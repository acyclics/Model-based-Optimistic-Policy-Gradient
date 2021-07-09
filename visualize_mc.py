import os
import numpy as np
import torch
from copy import deepcopy
from time import sleep
import matplotlib.pyplot as plt

import gym
import pybulletgym

from mbopg.mbrl_parallel import Policy


# Overall hyperparameters
N_OVERALL_POLICIES = 10
DEVICE = 'cpu'

# Setup environment variables
env = gym.make('MountainCarContinuous-v0')
#InvertedPendulumMuJoCoEnv, HalfCheetahMuJoCoEnv, InvertedPendulumSwingupPyBulletEnv, MountainCarContinuous, HopperPyBulletEnv, InvertedDoublePendulumPyBulletEnv
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high

# Setup policy
policy_dir = "./experiments/mc"

visited_states = []

for idx in range(N_OVERALL_POLICIES):
    policy_file = f"agent_loop{idx}"
    policy_file = os.path.join(policy_dir, policy_file)
    policy = Policy(None, action_range, 3, DEVICE, 0.4)
    policy.load(policy_file, cpu=True)

    current_state = env.reset()

    for _ in range(1000 * 10):
        greedy_u = policy.act(current_state, sample=True)
        next_state, rew, done, _ = env.step(greedy_u)
        current_state = next_state.copy()

        visited_states.append(current_state)

        if done:
            current_state = env.reset()

for visited_state in visited_states:
    x = visited_state[0]
    vel = visited_state[1]
    plt.scatter(x, vel, 2, 'red')

plt.show()
