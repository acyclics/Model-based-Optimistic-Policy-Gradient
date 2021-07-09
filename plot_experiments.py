import os
import torch
import numpy as np
import random
import argparse
from copy import deepcopy

import gym
import pybulletgym

from logs import _plot_line
from mbopg.mbrl import Policy


parser = argparse.ArgumentParser(description='mbopg_parallel')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--env', type=str, default='AntPyBulletEnv-v0', help='Environment name')
parser.add_argument('--n-explore-steps', type=int, default=1000, help='Number of explore steps')
parser.add_argument('--n-rollouts', type=int, default=10, help='Number of rollouts for evaluating agent')
parser.add_argument('--n-evals', type=int, default=100, help='Number of evals for evaluating agent')

DEVICE = 'cpu'
args = parser.parse_args()

base_dir = os.path.join(args.id, args.env)
all_seeds = os.listdir(base_dir)

metrics = {
    'steps': [step for step in range(args.n_explore_steps, args.n_rollouts * args.n_explore_steps, args.n_explore_steps)],
    'rewards': [[] for _ in range(args.n_rollouts)]
}

for seed in all_seeds:
    seed = seed[5:]
    seed = int(seed)
    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    agents_dir = os.path.join(seed_dir, 'agents')

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Setup environment variables
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high

    max_reward = -1000000000

    for idx in range(args.n_rollouts):
        try:
            policy_file = os.path.join(agents_dir, f"agent_loop{idx}")
            policy = Policy(None, action_range, 3, DEVICE)
            policy.load(policy_file, cpu=True)

            done = True
            T_rewards = []

            for _ in range(args.n_evals):
                while True:
                    if done:
                        state, reward_sum, done = env.reset(), 0, False

                    action = policy.eval_act(state, sample=True)
                    state, reward, done, info = env.step(action)
                    reward_sum += reward

                    if done:
                        T_rewards.append(reward_sum)
                        break

            avg_reward = sum(T_rewards) / len(T_rewards)
            max_reward = max(max_reward, avg_reward)
        except:
            pass
        metrics['rewards'][idx].append(max_reward)

_plot_line(metrics['steps'], metrics['rewards'], f"{args.env}: Max Average Reward", path=base_dir)
