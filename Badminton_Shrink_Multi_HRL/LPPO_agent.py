import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def to_one_hot(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_one_hot = np.zeros((y.shape[0], num_columns))
    y_one_hot[range(y.shape[0]), y] = 1.0

    return y_one_hot


def latent_uniform_sampling(latent_cont_dim, latent_disc_dim, sample_num):
    latent_dim = latent_cont_dim + latent_disc_dim
    z = None
    z_cont = None
    if not latent_cont_dim == 0:
        z_cont = np.random.uniform(-1, 1, size=(sample_num, latent_cont_dim))
        if latent_disc_dim == 0:
            z = z_cont
    if not latent_disc_dim == 0:
        z_disc = np.random.randint(0, latent_disc_dim, sample_num)
        z_disc = to_one_hot(z_disc, latent_disc_dim)
        if latent_cont_dim == 0:
            z = z_disc
        else:
            z = np.hstack((z_cont, z_disc))

    return z


class LatentGaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden=(128,64)):
        super().__init__()
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        ##default network setting
        self.l1 = nn.Linear(state_dim+latent_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], action_dim)
        ## Need 3 fully connected hidden layers each of which has 512 units ? 
        ###self.l1 = nn.Linear(state_dim + latent_dim,hidden[0])
        ###self.h1 = nn.Linear(hidden[0],hidden[0])
        ###self.h2 = nn.Linear(hidden[0],hidden[0])
        ###self.l2 = nn.Linear(hidden[0],action_dim)

    def _distribution(self, state, latent):
        ##print('state',state)
        ##print('latent',latent)
        sz = torch.cat([state, latent], 1)
        a = torch.tanh(self.l1(sz))
        a = torch.tanh(self.l2(a))
        mu = self.l3(a)
        ##a = torch.tanh(self.l1(sz))
        ##a = torch.tanh(self.h1(a))
        ##a = torch.tanh(self.h2(a))
        ##mu = torch.tanh(self.l2(a)) ## <- 活性化関数なしで良いのか？
        std = torch.exp(self.log_std)
        return Normal(mu, std), mu

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, state, latent, action=None):
        pi, a = self._distribution(state, latent)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(pi, action)
        return pi, logp_a


class LatentValue(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden=(128,64)):
        super(LatentValue, self).__init__()

        ##default network setting
        self.l1 = nn.Linear(state_dim+latent_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], 1)
        ###self.l1 = nn.Linear(state_dim + latent_dim,hidden[0])
        ###self.h1 = nn.Linear(hidden[0],hidden[0])
        ###self.h2 = nn.Linear(hidden[0],hidden[0])
        ###self.l2 = nn.Linear(hidden[0],1)

    def forward(self, state, latent):
        sz = torch.cat([state, latent], 1)
        h1 = torch.tanh(self.l1(sz))
        h1 = torch.tanh(self.l2(h1))
        v = self.l3(h1)
        ##b = torch.tanh(self.l1(sz))
        ##b = torch.tanh(self.h1(b))
        ##b = torch.tanh(self.h2(b))
        ##v = self.l2(b)

        return torch.squeeze(v, -1)


class LPPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, latent_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.latent_buf = np.zeros(combined_shape(size, latent_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, latent, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.latent_buf[self.ptr] = latent
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, latent=self.latent_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def sample_sa(self):
        data = dict(obs=self.obs_buf, act=self.act_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, latent_cont_dim=0, latent_disc_dim=0, hidden=(128,64)):
        super(Discriminator, self).__init__()

        # Z1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])

        if not latent_cont_dim == 0:
            self.l3_z_cont = nn.Linear(hidden[1], latent_cont_dim)

        if not latent_disc_dim == 0:
            self.l3_z_disc = nn.Linear(hidden[1], latent_disc_dim)

        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        z_cont = None
        z_disc = None
        h = F.relu(self.l1(sa))
        h = F.relu(self.l2(h))
        if not self.latent_cont_dim == 0:
            z_cont = self.l3_z_cont(h)
        if not self.latent_disc_dim == 0:
            z_disc = F.softmax(self.l3_z_disc(h))

        return z_cont, z_disc


class LPPO(object):
    def __init__(self,
            state_dim,
            action_dim,
            latent_cont_dim,
            latent_disc_dim,
            steps_per_epoch=4000,
            gamma=0.99,
            clip_ratio=0.2,
            pi_lr=3e-4,
            vf_lr=3e-4,
            hidden_sizes=(128,64),
            train_pi_iters=80,
            train_v_iters=80,
            lam=0.95,
            target_kl=0.01,
            iw = False
    ):
        latent_dim = latent_cont_dim + latent_disc_dim
        print('LatentGaussianActor : (state_dim,action_dim,latent_dim) = ',(state_dim,action_dim,latent_cont_dim))
        self.pi = LatentGaussianActor(state_dim, action_dim, latent_dim, hidden_sizes).to(device)

        # build value function
        self.value = LatentValue(state_dim, latent_dim, hidden_sizes).to(device)

        self.discriminator = Discriminator(state_dim, action_dim, latent_cont_dim=latent_cont_dim,
                                           latent_disc_dim=latent_disc_dim, hidden=(128,64)).to(device)

        self.clip_ratio = clip_ratio
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.latent_dim = latent_dim
        self.iw = iw

        # Set up optimizers for policy and value function
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=pi_lr)

        self.vf_optimizer = torch.optim.Adam(self.value.parameters(), lr=vf_lr)
        self.info_optimizer = torch.optim.Adam(itertools.chain(self.pi.parameters(),
                                                               self.discriminator.parameters()), lr=0.2*pi_lr)

        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

    def select_action(self, state, latent, stochastic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        latent = torch.FloatTensor(latent.reshape(1, -1)).to(device)
        a, v, logp, a_greedy = self.step(state, latent)
        if stochastic:
            return a.flatten(), v, logp
        return a_greedy.flatten()

    def step(self, obs, latent):
        with torch.no_grad():
            pi, a_greedy = self.pi._distribution(obs, latent)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.value(obs, latent)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), a_greedy.cpu().numpy()

    def act(self, obs, latent):
        return self.step(obs, latent)[0]

    def select_adv_latent(self, replay_buffer, batch_size=256, sample_num=50):
        z_min = None
        v_min = 1e6
        data = replay_buffer.sample_sa()
        obs = data['obs'].to(device)
        ind = np.random.randint(0, obs.shape[0], size=batch_size)
        state = obs[ind]

        for k in range(sample_num):
            z = self.sample_latent(self.latent_cont_dim, self.latent_disc_dim)
            z = torch.FloatTensor(z.reshape(1, -1)).to(device)
            z_tile = z.repeat(batch_size, 1)
            #action = torch.FloatTensor(action.reshape(1, -1)).to(device)
            #print('action', action.shape)
            v_tile = self.value(state, z_tile)
            v_mean = v_tile.mean().item()

            if v_min > v_mean:
                v_min = v_mean
                z_min = z

            # env_param_opt = self.recover_env_pram_z(z_min.cpu().data.numpy())

        return z_min.cpu().data.numpy()

    def sample_latent(self, latent_cont_dim, latent_disc_dim):

        z = None
        z_cont = None
        if not latent_cont_dim == 0:
            z_cont = np.random.uniform(-1, 1, size=(1, latent_cont_dim))
            if latent_disc_dim == 0:
                z = z_cont
        if not latent_disc_dim == 0:
            z_disc = np.random.randint(0, latent_disc_dim, 1)
            z_disc = self.to_one_hot(z_disc, latent_disc_dim)
            if latent_cont_dim == 0:
                z = z_disc
            else:
                z = np.hstack((z_cont, z_disc))

        return z

    def to_one_hot(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_one_hot = np.zeros((y.shape[0], num_columns))
        y_one_hot[range(y.shape[0]), y] = 1.0

        return y_one_hot

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, latent, adv, logp_old = data['obs'].to(device), data['act'].to(device), data['latent'].to(device), data['adv'].to(device), data['logp'].to(device)

        # Policy loss
        pi, logp = self.pi(obs, latent, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    #data = {"obs" : , "act" : , "latent" : , "adv" : , "logp" : ,} extract these from agent_trajectory_buffer
    def compute_info_loss(self, data):
        obs, act, latent, adv, logp_old = data['obs'].to(device), data['act'].to(device), data['latent'].to(device), data['adv'].to(device), data['logp'].to(device)

        latent_rand = torch.FloatTensor(
            latent_uniform_sampling(self.latent_cont_dim, self.latent_disc_dim, sample_num=obs.shape[0])).to(device)
        _, action_mu = self.pi._distribution(obs, latent_rand)
        z_cont, z_disc = self.discriminator(obs, action_mu)

        info_loss = 0
        if self.iw:
            disc_loss = nn.CrossEntropyLoss(reduction='none')
            with torch.no_grad():
                pi, logp = self.pi(obs, latent, act)
                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

            # print('clip_adv', clip_adv.shape)

            if not self.latent_cont_dim == 0:
                latent_cont = None
                if self.latent_disc_dim == 0:
                    latent_cont = latent
                else:
                    latent_cont = latent[:, 0:self.latent_cont_dim]

                info_loss += torch.mean(
                    clip_adv.detach().view(-1, 1) * F.mse_loss(z_cont, latent_cont.detach(), reduction='none'))

            if not self.latent_disc_dim == 0:
                latent_disc = None
                if self.latent_cont_dim == 0:
                    latent_disc = latent
                else:
                    latent_disc = latent[:, self.latent_cont_dim:self.latent_dim]

                latent_disc_label = torch.argmax(latent_disc, dim=1)
                info_loss += torch.mean(
                    clip_adv.detach().view(-1, 1) * disc_loss(z_disc, latent_disc_label.detach()).view(-1, 1))

        else:
            if not self.latent_cont_dim == 0:
                latent_cont = None
                if self.latent_disc_dim == 0:
                    latent_cont = latent_rand
                else:
                    latent_cont = latent_rand[:, 0:self.latent_cont_dim]
                info_loss += F.mse_loss(z_cont, latent_cont)

            if not self.latent_disc_dim == 0:
                latent_disc = None
                if self.latent_cont_dim == 0:
                    latent_disc = latent_rand
                else:
                    latent_disc = latent_rand[:, self.latent_cont_dim:self.latent_dim]

                latent_disc_label = torch.argmax(latent_disc, dim=1)
                info_loss += F.cross_entropy(z_disc, latent_disc_label)

        return info_loss

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, latent, ret = data['obs'].to(device), data['latent'].to(device), data['ret'].to(device)
        return ((self.value(obs, latent) - ret) ** 2).mean()

    def update(self, buf):
        data = buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)

            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step ', i,' due to reaching max kl.')
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()

            self.info_optimizer.zero_grad()
            loss_info = self.compute_info_loss(data)

            loss_info.backward()
            self.info_optimizer.step()


        # logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

    def save_model(self, iter, seed, env_name, foldername='./models/lppo/low_level'):
        try:
            import pathlib
            pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)

            torch.save(self.pi.state_dict(),
                       foldername + '/lppo_actor_' + env_name + '_seed' + str(seed) + '_iter' + str(iter) + '.pth')

            torch.save(self.value.state_dict(),
                       foldername + '/lppo_value_' + env_name + '_seed' + str(seed) + '_iter' + str(iter) + '.pth')

            print('models is saved for iteration', iter)

        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")

    def load_model(self, iter, seed, env_name, foldername='models/lppo'):

        self.pi.load_state_dict(torch.load(
            foldername + '/lppo_actor_' + env_name + '_seed' + str(seed) + '_iter' + str(iter) + '.pth',map_location=device))

        self.value.load_state_dict(torch.load(
            foldername + '/lppo_value_' + env_name + '_seed' + str(seed) + '_iter' + str(iter) + '.pth',map_location=device))


