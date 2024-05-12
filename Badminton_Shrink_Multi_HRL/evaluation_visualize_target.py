import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from gym.spaces import Box
import random

import pprint as pp

from PPO_agent import PPO, PPOBuffer
from LPPO_agent import LPPO,LPPOBuffer
##from PPO_HRL_agent import PPO,PPOBuffer

from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

# import spinup.algos.pytorch.ppo.core as core
# from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def transform_high_level(high_level_action):
    high_level_action = np.clip(high_level_action,-1.0,1.0)
    theta = high_level_action[0] * 5 + 45
    phi = high_level_action[1] * 30
    r = high_level_action[2] * 1.5 + 3

    y = r * np.cos(np.deg2rad(theta))
    x = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    z = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))

    return np.array([x,y,z])

def evaluate_greedy_high_level(env_test, high_level_agent,low_level_agent, args, test_iter, test_n, state_dim):
    ##print('evaluation start')
    states_test = env_test.reset()
    obs_dict = {}
    high_level_actions = {}
    low_level_actions = {}
    ##state_test = env_test.reset()
    returns_epi_test = {agent : 0 for agent in env_test.agents}
    return_epi_test = 0

    while True : 
        for t_test in range(int(args['max_episode_len'])):
            for agent in env_test.agents:
                obs_dict[agent] = states_test[agent]
            if t_test % 10 == 0:
                ##print('high level action sampled')
                high_level_actions = {agent : high_level_agent.select_action(np.array(obs_dict[agent])) for agent in env_test.agents}
            low_level_actions = {agent : low_level_agent.select_action(np.reshape(obs_dict[agent], (1, state_dim)),high_level_actions[agent]) for agent in env_test.agents}
            for agent in env_test.agents:
                lower_bound = -1.0
                upper_bound = 1.0
                low_level_actions[agent] = np.clip(low_level_actions[agent],lower_bound,upper_bound)
            ##print('low-level action sampled')
            ##action_test = high_level_agent.select_action(np.reshape(state_test, (1, state_dim)))
            ##state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
            new_states_test,rewards_test,terminals_test,infos_test = env_test.step(low_level_actions)
            ##state_test = state_test2
            states_test = new_states_test
            for agent in env_test.agents:
                returns_epi_test[agent] += rewards_test[agent]
        
            ##return_epi_test = return_epi_test + reward_test
            if any([terminals_test[a] for a in terminals_test]):
                print('evaluation came to terminal state')
                env_test.reset()
                break

        for agent in env_test.agents:
            print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      int(returns_epi_test[agent])))


def evaluate_greedy_low_level(env_test, low_level_agent, args, test_iter, test_n, state_dim, latent_cont_dim, latent_disc_dim):

    states_test = env_test.reset()
    ##state_test = env_test.reset()
    agent_to_evaluate = env_test.agents[0]
    low_level_actions = {}

    z = uniform_sampling(latent_cont_dim, latent_disc_dim)
    while True : 
        return_epi_test = 0
        for t_test in range(int(args['max_episode_len'])):
            ##print('states : ',states_test[agent_to_evaluate])
            selected_actions = {agent : low_level_agent.select_action(np.reshape(states_test[agent], (1, state_dim)), z, stochastic=False) for agent in env_test.agents}
            ##print('selected action : ',selected_actions[agent_to_evaluate])
            for agent in env_test.agents:
                lower_bound = -1.0
                upper_bound = 1.0
                selected_actions[agent] = np.clip(selected_actions[agent],lower_bound,upper_bound)
            ##print('selected action',selected_actions[agent_to_evaluate])
        ##action_test = agent.select_action(np.reshape(state_test, (1, state_dim)), z, stochastic=False)
        ##state_test2, reward_test, terminal_test, info_test = env_test.step(selected_actions)
            new_states, rewards_test,terminals_test,infos_test = env_test.step(selected_actions)
            ##print('info_test : ',infos_test)
            states_test = new_states
            return_epi_test = return_epi_test + rewards_test[agent_to_evaluate]
            if any([terminals_test[a] for a in terminals_test]):
                print('step',t_test)
                env_test.reset()
                break

        """print('LOW_LEVEL : test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      int(return_epi_test)))"""
        print('| LOW_LEVEL | test_iter: ',test_iter,'| nn: ',test_n,' |  return_epi_test: ',return_epi_test,' |')

def evaluate_greedy_ppo(env_test, low_level_agent, args, test_iter, test_n, state_dim):
    print('state dim : ',state_dim)
    states_test = env_test.reset()
    print('agents : ',env_test.agents)
    agent1 = env_test.agents[0]
    agent2 = env_test.agents[1]
    low_level_actions = {}

    ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)
    while True : 
        return_epi_test = 0
        for t_test in range(int(args['max_episode_len'])):
            selected_actions = {agent : low_level_agent.select_action(np.reshape(states_test[agent], (1, state_dim)), stochastic=False) for agent in env_test.agents}
            for agent in env_test.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                selected_actions[agent] = np.clip(selected_actions[agent],lower_bound,upper_bound)
            print('action : ',selected_actions[agent1])
            new_states, rewards_test,terminals_test,infos_test = env_test.step(selected_actions)
            states_test = new_states
            return_epi_test = return_epi_test + rewards_test[agent1]
            if any([terminals_test[a] for a in terminals_test]):
                print('step',t_test)
                env_test.reset()
                break
        print('| LOW_LEVEL | test_iter: ',test_iter,'| nn: ',test_n,' |  return_epi_test: ',return_epi_test,' |')

def evaluate_greedy_high_random(env_test,low_level_agent, args, test_iter, test_n, state_dim):
    ##print('evaluation start')
    while True:
        states_test = env_test.reset()
        agent_to_train = env_test.agents[0]
        obs_dict = {}
        low_level_states = {}
        high_level_flags = {agent : False for agent in env_test.agents}
        high_level_actions = {agent : 0.0 for agent in env_test.agents}
        low_level_actions = {}
        ##state_test = env_test.reset()
        return_epi_test = 0
        for t_test in range(int(args['max_episode_len'])):
            for agent in env_test.agents:
                obs_dict[agent] = states_test[agent]
                low_level_states[agent] = states_test[agent][:17]
            for agent in env_test.agents:
                if obs_dict[agent][19] > 0.5 and high_level_flags[agent] == False:
                    ##high_level_actions[agent] = high_level_agent.select_action(np.array(obs_dict[agent]),stochastic = True)
                    sample_high_level = uniform_sampling(1,0)
                    target_position = 4.0 + 2.0 * sample_high_level[0][0]
                    high_level_actions[agent] = target_position
                    print('target_position',target_position)
                    ##high_level_actions[agent] = transform_high_level(uniform_sampling)
                    print(agent,'sample action')
                low_level_states[agent][16] = high_level_actions[agent] ##最後の三つの部分を置き換える。
                high_level_flags[agent] = obs_dict[agent][19] > 0.5
            ##print('state : ',low_level_states[agent_to_train])
            low_level_actions = {agent : low_level_agent.select_action(np.reshape(low_level_states[agent], (1, state_dim)),stochastic=False) for agent in env_test.agents}
            for agent in env_test.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                low_level_actions[agent] = np.clip(low_level_actions[agent],lower_bound,upper_bound)
            
            new_states_test,rewards_test,terminals_test,infos_test = env_test.step(low_level_actions)
            states_test = new_states_test
            return_epi_test += rewards_test[agent_to_train]
        
            if any([terminals_test[a] for a in terminals_test]):
                print('evaluation came to terminal state')
                env_test.reset()
                break

        for agent in env_test.agents:
            print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      int(return_epi_test)))
            
def evaluate_high(env_test,high_level_agent,low_level_agent, args, test_iter, test_n, state_dim):
    ##print('evaluation start')
    episode_count = 0
    total_rally = 0
    while True:
        states_test = env_test.reset()
        agent_to_train = env_test.agents[0]
        obs_dict = {}
        low_level_states = {}
        high_level_flags = {agent : False for agent in env_test.agents}
        high_level_actions = {agent : 0.0 for agent in env_test.agents}
        low_level_actions = {}
        ##state_test = env_test.reset()
        return_epi_test = 0
        for t_test in range(int(args['max_episode_len'])):
            for agent in env_test.agents:
                obs_dict[agent] = states_test[agent]
                low_level_states[agent] = states_test[agent][:17]
            for agent in env_test.agents:
                if obs_dict[agent][19] > 0.5 and high_level_flags[agent] == False:
                    sample_high_level = high_level_agent.select_action(np.array(obs_dict[agent]),stochastic = False)
                    ##sample_high_level = uniform_sampling(1,0)
                    target_position = 4.0 + 2.0 * sample_high_level[0]
                    high_level_actions[agent] = target_position
                    ##print('target_position',target_position)
                    ##high_level_actions[agent] = transform_high_level(uniform_sampling)
                    ##print(agent,'sample action')
                low_level_states[agent][16] = high_level_actions[agent] ##最後の三つの部分を置き換える。
                high_level_flags[agent] = obs_dict[agent][19] > 0.5
            ##print('state : ',low_level_states[agent_to_train])
            low_level_actions = {agent : low_level_agent.select_action(np.reshape(low_level_states[agent], (1, state_dim)),stochastic=False) for agent in env_test.agents}
            for agent in env_test.agents:
                low_level_actions[agent] = np.concatenate((low_level_actions[agent],np.array([high_level_actions[agent]])),axis=0)

            print('output action : ',low_level_actions[agent_to_train])
            for agent in env_test.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                low_level_actions[agent] = np.clip(low_level_actions[agent],lower_bound,upper_bound)
            
            new_states_test,rewards_test,terminals_test,infos_test = env_test.step(low_level_actions)
            states_test = new_states_test
            return_epi_test += rewards_test[agent_to_train]
        
            if any([terminals_test[a] for a in terminals_test]):
                print('evaluation came to terminal state')
                total_rally += return_epi_test
                print('average rally length : ',total_rally / episode_count)
                episode_count += 1
                env_test.reset()
                break

        for agent in env_test.agents:
            print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      int(return_epi_test)))

def evaluate_manual_input(env_test,low_level_agent, args, test_iter, test_n, state_dim, latent_cont_dim, latent_disc_dim):
    states_test = env_test.reset()
    ##state_test = env_test.reset()
    agent_to_evaluate = env_test.agents[0]
    low_level_actions = {}
    random_actions = {agent : env_test.action_space(agent).sample() for agent in env_test.agents}
    print('random_actions',random_actions)
    sample_1 = np.array([-0.753, -0.10475154, -0.4967119 , -0.4227658 , -0.44580758,
        0.66514057, -0.8002908 , 0 ,  0.6518614 ,  0.67212766,
       -0.5344978 ,  0.43816134, -0.2323 , 1.0  ,  0.05433765,
        0.9678086 ])
    ##-0.38905042
    z = uniform_sampling(latent_cont_dim, latent_disc_dim)
    index = 0
    return_epi_test = 0
    zeros = np.zeros((16,),dtype=np.float32)
    ones = np.ones((16,),dtype=np.float32)
    halves = np.full(16,0.5,dtype=np.float32)
    actions = sample_1
    actions[0] = 0
    ##actions[0] = 0.0
    zeros[index] = 1.0
    count = 0
    while True : 
        for t_test in range(int(args['max_episode_len'])):
            count += 1
            ##print('count',count)
            """if count == 5 and actions[index] == -1.0:
                actions[index] = 1.0
                count = 0
                print('zeros',actions)
            elif count == 5 and actions[index] == 1.0:
                actions[index] = -1.0
                count = 0
                print('zeros',actions)"""
            ##selected_actions = {agent : sample_1 for agent in env_test.agents}
            ##selected_actions = {agent : env_test.action_space(agent).sample() for agent in env_test.agents}
            ##actions[0] = random.uniform(-1.0,1.0)
            selected_actions = {agent : actions for agent in env_test.agents}
            ##print('selected actions : ',selected_actions[agent_to_evaluate])
            new_states, rewards_test,terminals_test,infos_test = env_test.step(selected_actions)
            states_test = new_states
            return_epi_test = return_epi_test + rewards_test[agent_to_evaluate]
            if any([terminals_test[a] for a in terminals_test]):
                env_test.reset()
                break

def to_1d_list(state):
    single_d = [item for sublist in state for item in sublist]
    return single_d

def to_one_hot(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_one_hot = np.zeros((y.shape[0], num_columns))
    y_one_hot[range(y.shape[0]), y] = 1.0

    return y_one_hot

def uniform_sampling(latent_cont_dim, latent_disc_dim):
    z = None
    z_cont = None
    if not latent_cont_dim == 0:
        z_cont = np.random.uniform(-1, 1, size=(1, latent_cont_dim))
        if latent_disc_dim == 0:
            z = z_cont
    if not latent_disc_dim == 0:
        z_disc = np.random.randint(0, latent_disc_dim, 1)
        z_disc = to_one_hot(z_disc, latent_disc_dim)
        if latent_cont_dim == 0:
            z = z_disc
        else:
            z = np.hstack((z_cont, z_disc))

    return z
## use only one replay buffer

def main(args):
    for ite in range(int(args['trial_num'])):
        print('Trial Number:', ite)
        ##env = gym.make(args['env'])

        unity_low_level_env = UnityEnvironment(args['high_level_env'],no_graphics=False,worker_id=int(args['port_offset_low']))
        low_level_env = UnityParallelEnv(unity_low_level_env)
        low_level_env.seed(int(args['random_seed']))
        low_level_env_test = UnityParallelEnv(unity_low_level_env)
        low_level_env_test.seed(int(args['random_seed']))

        agent_names = low_level_env.agents

        assert (low_level_env.action_space(agent_names[0]).high[0] == -low_level_env.action_space(agent_names[0]).low[0])

        state_dim = low_level_env.observation_space(agent_names[0]).shape[0]
        action_dim = low_level_env.action_space(agent_names[0]).shape[0]
        latent_cont_dim = int(args['latent_cont_dim'])
        latent_dist_dim = int(args['latent_disc_dim'])

        print('action_space.shape', state_dim)
        print('observation_space.shape', action_dim)
        action_bound = float(low_level_env.action_space(agent_names[0]).high[0])

        """high_level_agent = PPO(state_dim=state_dim, action_dim=latent_cont_dim+latent_dist_dim,
                    ac_kwargs=dict(hidden_sizes=[args['hid']]*args['l']))"""
        high_level_agent = PPO(20,1,ac_kwargs=dict(hidden_sizes=(64,64)))
        high_level_agent.load_model(args['iter_high'],args['seed_high'],args['high_level_env'])
        
        ##low_level_agent = LPPO(state_dim=state_dim, action_dim=action_dim, ##load low leve state dict after the low-level training
                     ##latent_cont_dim=latent_cont_dim, latent_disc_dim=0)
        low_level_agent = PPO(17,3,ac_kwargs=dict(hidden_sizes=(64,64)))
        low_level_agent.load_model(args['iter'],args['seed'],args['low_level_env'])
        ##evaluate_greedy_ppo(low_level_env_test,low_level_agent,args,0,0,state_dim)
        if int(args['train_high_level']) == 0:                
            print('Only Low-leve')
            evaluate_greedy_high_random(low_level_env_test,low_level_agent,args,0,0,17)
        else:
            print('HRL')
            evaluate_high(low_level_env_test,high_level_agent,low_level_agent,args,0,0,17)
            ##evaluate_manual_input(low_level_env_test,low_level_agent,args,0,0,state_dim,latent_cont_dim,latent_dist_dim)
            ##low_level_agent.load_model(args["iter"],args["seed"],args["env"])
        print("Low-Level Training Finished")
            ## if train_high_level == True, run high-leve training


            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--high_level_env', help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--low_level_env')
    parser.add_argument('--env-id', type=int, default=6, help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument("--latent-cont-dim", default=2, type=int)  # dimension of the continuous latent variable
    parser.add_argument("--latent-disc-dim", default=0, type=int)  # dimension of the discrete latent variable

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--total-step-num', help='total number of time steps', default=1000000)
    parser.add_argument('--eval-step-freq', help='frequency of evaluating the policy', default=5000)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=1)
    parser.add_argument('--test-num', help='number of test episodes', default=10)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=3)
    parser.add_argument('--model-save-freq', help='frequency of evaluating the policy', default=200000)
    parser.add_argument('--result_file',default='/hrl_')

    parser.add_argument('--iter',default = 1001)
    parser.add_argument('--seed',default=13)
    parser.add_argument('--iter_high',default=1)
    parser.add_argument('--seed_high',default=13)
    parser.add_argument('--port_offset_high',default=0)
    parser.add_argument('--port_offset_low',default = 100)
    parser.add_argument('--check_point_path_low_level')
    parser.add_argument('--check_point_path_high_level')
    parser.add_argument('--train_high_level',type=int,default=0)
    args = parser.parse_args()

    args_tmp = parser.parse_args()

    """if args_tmp.env is None:
        env_dict = {00 : "Pendulum-v0",
                    1 : "InvertedPendulum-v1",
                    2 : "InvertedDoublePendulum-v1",
                    3 : "Reacher-v3",
                    4 : "Swimmer-v3",
                    5 : "Ant-v3",
                    6 : "Hopper-v3",
                    7 : "Walker2d-v3",
                    8 : "HalfCheetah-v3",
                    9 : "Humanoid-v3",
                    }
        args_tmp.env = env_dict[args_tmp.env_id]"""
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)