import numpy as np
import torch
from copy import deepcopy
from time import sleep

import gym
import pybulletgym

from mbopg.mbrl_parallel import Policy


# Overall hyperparameters
N_OVERALL_LOOPS = 1000
DEVICE = 'cpu'

# Setup environment variables
env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0')
#InvertedPendulumMuJoCoEnv, HalfCheetahMuJoCoEnv, InvertedPendulumSwingupPyBulletEnv, InvertedDoublePendulumPyBulletEnv
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high

# Setup policy
policy_file = "./agents/agent_loop7"
policy = Policy(None, action_range, 1, DEVICE)
policy.load(policy_file, cpu=True)
#controls = np.load(policy_file, allow_pickle=True)

for loop_n in range(N_OVERALL_LOOPS):
    print(f"Overall loop {loop_n}")

    done = False
    total_reward = 0
    total_steps = 0
    env.render()
    current_state = env.reset()
    #env._max_episode_steps = 100000

    while not done:
    #for control in controls:
        greedy_u = policy.act(current_state, sample=True)
        #print(greedy_u)
        #greedy_u = env.action_space.sample()
        next_state, rew, done, _ = env.step(greedy_u)
        #print(next_state)
        #print(rew)

        #err = 0.5 * (np.mean((next_state - current_state)**2))
        #print(err)

        total_reward += rew
        total_steps += 1
        current_state = next_state.copy()

        sleep(0.01)
    
    print(f"Total reward = {total_reward} ; Total steps = {total_steps}")
