import numpy as np

import pprint as pp

from PPO_agent import PPO

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


def evaluate_greedy_ppo(env_test, agent, args, test_iter, state_dim):
    state_test = env_test.reset()
    ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)

    for i in range(test_iter):
        return_epi_test = 0
        for t_test in range(int(args["max_episode_len"])):
            ##print('state_test',state_test)
            action_test = agent.select_action(
                np.reshape(state_test, (1, state_dim)), stochastic=False
            )
            state_test2, reward_test, terminal_test, info_test = env_test.step(
                action_test
            )
            state_test = state_test2
            return_epi_test = return_epi_test + reward_test
            if terminal_test:
                env_test.reset()
                break

        print(
            "| test_iter: ",
            i + 1,
            " | return_epi_test: ",
            return_epi_test,
            " |",
        )
        return_epi_test = 0


def main(args):
    unity_env = UnityEnvironment(args["env"])
    test_env = UnityToGymWrapper(unity_env)

    np.random.seed(int(args["random_seed"]))
    test_env.seed(int(args["random_seed"]))

    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]

    agent = PPO(state_dim, action_dim, ac_kwargs=dict(hidden_sizes=(64, 64)))

    agent.load_model(args["iter"], args["seed"], args["env"])
    evaluate_greedy_ppo(test_env, agent, args, int(args["test_num"]), state_dim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="choose the gym env- tested on {Pendulum-v0}")
    parser.add_argument(
        "--max-episode-len", help="max length of 1 episode", default=1000
    )
    parser.add_argument(
        "--total-step-num", help="total number of time steps", default=1000000
    )
    parser.add_argument(
        "--eval-step-freq", help="frequency of evaluating the policy", default=5000
    )
    parser.add_argument("--trial-num", help="number of trials", default=1)
    parser.add_argument("--trial-idx", help="index of trials", default=13)
    parser.add_argument("--test-num", help="number of test episodes", default=10)
    parser.add_argument(
        "--random-seed", help="random seed for repeatability", default=3
    )
    parser.add_argument("--iter", help="iteration number", default=1)
    parser.add_argument("--seed", help="training seed", default=601)
    args = parser.parse_args()

    args_tmp = parser.parse_args()
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
