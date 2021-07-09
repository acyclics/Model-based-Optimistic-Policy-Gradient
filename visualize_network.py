import numpy as np
import random
import torch
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

import gym
import pybulletgym

from mbopg.bh_mdp import Multi_SI_BayesNetwork
from mbopg.mbrl_parallel import Policy
#from rl.mdp_solver.mdp_solver_psrl_deter import Policy


# Overall hyperparameters
N_OVERALL_LOOPS = 100
DEVICE = 'cpu'

# Setup environment variables
env = gym.make('MountainCarContinuous-v0')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high

# Setup policy
#policy_file = "./agents/agent_loop21_p1"
weight_file = "./experiments/hypernetwork_model"
#model_for_InvertedPendulumSwingupPyBulletEnv-v0_experiment_boss_regparallel
#policy = Policy(None, action_range, DEVICE)
#policy.load(policy_file, cpu=True)
#w = torch.load(weight_file, map_location=torch.device('cpu'))
#controls = np.load(policy_file, allow_pickle=True)
REWARD_RANGE = [-2, 2]
network = Multi_SI_BayesNetwork(obs_dim, action_dim, [1], 256, 5, noise_dim=3,
                                grad_norm=5.0, LLH_var=1.0, device=DEVICE)
network.load(weight_file, 'cpu')
w = network.sample(1)

for loop_n in range(N_OVERALL_LOOPS):
    print(f"Overall loop {loop_n}")

    done = False
    total_reward = 0
    env.render()
    current_state = env.reset()
    net_current_state = current_state.copy()
    net_current_state = torch.from_numpy(net_current_state).float().unsqueeze(dim=0)

    # TEST
    z = (-2.0 - 2.0) * torch.rand((1, 3)) + 2.0
    z.requires_grad=True
    
    for _ in range(3):
        greedy_u = env.action_space.sample()
        current_state, rew, done, _ = env.step(greedy_u)
        pred_next_state, pred_rew, _ = network.primarynet(net_current_state, torch.from_numpy(greedy_u).float().unsqueeze(dim=0), w)
        net_current_state = pred_next_state[:, 0]
        print("Pred")
        print(net_current_state)
        print("Actual")
        print(current_state)
        print()
        print()
    
    #net_current_state = current_state.copy()
    #net_current_state = torch.from_numpy(net_current_state).float().unsqueeze(dim=0)
    greedy_u = env.action_space.sample()

    while True:
        w = network.sample(1, z)
        pred_next_state, pred_rew, _ = network.primarynet(net_current_state, torch.from_numpy(greedy_u).float().unsqueeze(dim=0), w)
        z_grad = torch.autograd.grad(torch.mean(pred_rew), z)
        print(pred_rew, z_grad, z)
        #z = torch.add(z, )
        z = z + z_grad[0] * 0.1
    # TEST

    for _ in range(1):
    #for control in controls:
        #greedy_u = policy.act(current_state, sample=True)
        greedy_u = env.action_space.sample()
        #greedy_u += np.random.randn((6)) * 0.5
        #greedy_u = control
        next_state, rew, done, _ = env.step(greedy_u)
        pred_next_state, _, pred_rew, _ = network.primarynet(net_current_state, torch.from_numpy(greedy_u).float().unsqueeze(dim=0), w)

        print("Actual next state:")
        print(next_state)

        print("Pred next state:")
        print(pred_next_state)

        total_reward += rew
        net_current_state = pred_next_state[:, 0]
        current_state = next_state.copy()
    
    print(f"Total reward = {total_reward}")
