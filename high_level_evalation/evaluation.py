import numpy as np

import pprint as pp

from PPO_agent import PPO
##from PPO_HRL_agent import PPO,PPOBuffer

from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv


def transform_high_level(high_level_action):
    high_level_action = np.clip(high_level_action, -1.0, 1.0)
    theta = high_level_action[0] * 5 + 45
    phi = high_level_action[1] * 30
    r = high_level_action[2] * 1.5 + 3

    y = r * np.cos(np.deg2rad(theta))
    x = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    z = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))

    return np.array([x, y, z])


def evaluate_greedy_high_random(
    env_test, low_level_agent, args, test_iter, test_n, state_dim
):
    ##print('evaluation start')
    episode_count = 1
    total_rally = 0
    success_count = 0
    while True:
        states_test = env_test.reset()
        agent_to_train = env_test.agents[0]
        obs_dict = {}
        low_level_states = {}
        high_level_flags = {agent: False for agent in env_test.agents}
        high_level_actions = {agent: 0.0 for agent in env_test.agents}
        low_level_actions = {}
        ##state_test = env_test.reset()
        return_epi_test = 0
        for t_test in range(int(args["max_episode_len"])):
            for agent in env_test.agents:
                obs_dict[agent] = states_test[agent]
                low_level_states[agent] = states_test[agent][:17]
            for agent in env_test.agents:
                if obs_dict[agent][19] > 0.5 and high_level_flags[agent] == False:
                    ##high_level_actions[agent] = high_level_agent.select_action(np.array(obs_dict[agent]),stochastic = True)
                    sample_high_level = uniform_sampling(1, 0)
                    target_position = 4.0 + 2.0 * sample_high_level[0][0]
                    high_level_actions[agent] = target_position
                    ##print('target_position',target_position)
                    ##high_level_actions[agent] = transform_high_level(uniform_sampling)
                    ##print(agent,'sample action')
                low_level_states[agent][16] = high_level_actions[
                    agent
                ]  ##最後の三つの部分を置き換える。
                high_level_flags[agent] = obs_dict[agent][19] > 0.5
            ##print('state : ',low_level_states[agent_to_train])
            low_level_actions = {
                agent: low_level_agent.select_action(
                    np.reshape(low_level_states[agent], (1, state_dim)),
                    stochastic=False,
                )
                for agent in env_test.agents
            }
            for agent in env_test.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                low_level_actions[agent] = np.clip(
                    low_level_actions[agent], lower_bound, upper_bound
                )

            new_states_test, rewards_test, terminals_test, infos_test = env_test.step(
                low_level_actions
            )
            states_test = new_states_test
            return_epi_test += rewards_test[agent_to_train]

            if any([terminals_test[a] for a in terminals_test]):
                print("evaluation came to terminal state")
                if return_epi_test >= 10.0:
                    success_count += 1
                total_rally += return_epi_test
                print(
                    "Episode : ",
                    episode_count,
                    "average rally length : ",
                    total_rally / episode_count,
                    "Sucess Count : ",
                    success_count,
                )
                episode_count += 1
                env_test.reset()
                break

        for agent in env_test.agents:
            print(
                "test_iter:{:d}, nn:{:d}, return_epi_test: {:d}".format(
                    int(test_iter), int(test_n), int(return_epi_test)
                )
            )


def evaluate_high(
    env_test, high_level_agent, low_level_agent, args, test_iter, state_dim
):
    ##print('evaluation start')
    episode_count = 1
    total_rally = 0
    success_count = 0
    for i in range(test_iter):
        states_test = env_test.reset()
        agent_to_train = env_test.agents[0]
        obs_dict = {}
        low_level_states = {}
        high_level_flags = {agent: False for agent in env_test.agents}
        high_level_actions = {agent: 0.0 for agent in env_test.agents}
        low_level_actions = {}
        ##state_test = env_test.reset()
        return_epi_test = 0
        for t_test in range(int(args["max_episode_len"])):
            for agent in env_test.agents:
                obs_dict[agent] = states_test[agent]
                low_level_states[agent] = states_test[agent][:17]
            for agent in env_test.agents:
                if obs_dict[agent][19] > 0.5 and high_level_flags[agent] == False:
                    sample_high_level = high_level_agent.select_action(
                        np.array(obs_dict[agent]), stochastic=False
                    )
                    ##sample_high_level = uniform_sampling(1,0)
                    target_position = 4.0 + 2.0 * sample_high_level[0]
                    high_level_actions[agent] = target_position
                    ##print('target_position',target_position)
                    ##high_level_actions[agent] = transform_high_level(uniform_sampling)
                    ##print(agent,'sample action')
                low_level_states[agent][16] = high_level_actions[
                    agent
                ]  ##最後の三つの部分を置き換える。
                high_level_flags[agent] = obs_dict[agent][19] > 0.5
            ##print('state : ',low_level_states[agent_to_train])
            low_level_actions = {
                agent: low_level_agent.select_action(
                    np.reshape(low_level_states[agent], (1, state_dim)),
                    stochastic=False,
                )
                for agent in env_test.agents
            }
            for agent in env_test.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                low_level_actions[agent] = np.clip(
                    low_level_actions[agent], lower_bound, upper_bound
                )

            new_states_test, rewards_test, terminals_test, infos_test = env_test.step(
                low_level_actions
            )
            states_test = new_states_test
            return_epi_test += rewards_test[agent_to_train]

            if any([terminals_test[a] for a in terminals_test]):
                print("evaluation came to terminal state")
                if return_epi_test >= 10.0:
                    success_count += 1
                total_rally += return_epi_test / 0.2
                ##print('Episode : ',episode_count,'average rally length : ',total_rally / episode_count,'Sucess Count : ',success_count)
                print("Rally Length: ", return_epi_test)
                episode_count += 1
                env_test.reset()
                break

        print("test_iter:{:d}, return_epi_test: {:d}".format(i, int(return_epi_test)))


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
    ##env = gym.make(args['env'])

    unity_low_level_env = UnityEnvironment(args["high_level_env"], no_graphics=False)
    low_level_env = UnityParallelEnv(unity_low_level_env)
    low_level_env.seed(int(args["random_seed"]))
    low_level_env_test = UnityParallelEnv(unity_low_level_env)
    low_level_env_test.seed(int(args["random_seed"]))

    agent_names = low_level_env.agents

    assert (
        low_level_env.action_space(agent_names[0]).high[0]
        == -low_level_env.action_space(agent_names[0]).low[0]
    )

    state_dim = low_level_env.observation_space(agent_names[0]).shape[0]
    action_dim = low_level_env.action_space(agent_names[0]).shape[0]

    print("action_space.shape", state_dim)
    print("observation_space.shape", action_dim)
    high_level_agent = PPO(20, 1, ac_kwargs=dict(hidden_sizes=(64, 64)))
    high_level_agent.load_model(
        args["iter_high"], args["seed_high"], args["high_level_env"]
    )
    low_level_agent = PPO(17, 3, ac_kwargs=dict(hidden_sizes=(64, 64)))
    low_level_agent.load_model(
        args["iter_low"], args["seed_low"], args["low_level_env"]
    )
    evaluate_high(
        low_level_env_test,
        high_level_agent,
        low_level_agent,
        args,
        int(args["test_num"]),
        17,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--high_level_env", default="High_level")
    parser.add_argument("--low_level_env", default="Low_level")

    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--test-num", help="number of test episodes", default=5)
    parser.add_argument(
        "--random-seed", help="random seed for repeatability", default=3
    )
    parser.add_argument("--max-episode-len", default=1000)

    parser.add_argument("--iter_low", default=601)
    parser.add_argument("--seed_low", default=1)
    parser.add_argument("--iter_high", default=26)
    parser.add_argument("--seed_high", default=1)
    args = parser.parse_args()

    args_tmp = parser.parse_args()
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
