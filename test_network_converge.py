import sys
import numpy as np
import random
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

import gym
import pybulletgym

from mbopg.mbrl_parallel import Policy
from mbopg.bh_mdp import Multi_SI_BayesNetwork


# Overall hyperparameters
DEVICE = 'cuda'
N_OVERALL_LOOPS = 250
# Network parameters
BETA_THRESHOLD = 100
BETA_STEPSIZE = 1.0
DATASET_BATCH_SIZE = 1000
K_VALUE = 1
J_VALUE = 1
EPOCHS_PER_DATASET_ITR = 100000
N_HIDDEN_UNITS = 512
N_HIDDEN_LAYERS = 5
GRAD_NORM = 5.0
# Model-based parameters
N_STEPS_PER_RL = 3
N_STEPS_PER_EXPLORATION = 100000
MDP_EPOCHS = 2000
ACTOR_ITRS_PER_EPOCH = 3
CRITIC_ITRS_PER_EPOCH = 5
CRITIC_BATCH_SIZE = 32
CAPACITY = 1e5
# Actor/Critic parameters
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GAMMA = 0.99
ALPHA = 0.1
TARGET_UPDATE_FREQ = 2
# IDDPS parameters
N_MDPS = 5
ROLLOUT_ITRS = 5

# Setup environment variables
REWARD_RANGE = [-2, 2]
env = gym.make('HopperPyBulletEnv-v0')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high
actor_logstd_bounds = [-5, 2]
actor_hidden_dim = 64
actor_hidden_layers = 3
critic_hidden_dim = 64
critic_hidden_layers = 3

#policy_file = "./agents/boss_regparallel/InvertedPendulumSwingupPyBulletEnv-v0/agent_loop0"
#policy_file = "./agents/agent_loop0"
policy = Policy(None, action_range, 1, DEVICE)
#policy.load(policy_file)

# Setup network
PATH = "./implicit/multi_semi_implicit/throwaway"
network = Multi_SI_BayesNetwork(obs_dim, action_dim, [10 for _ in range(15)], N_HIDDEN_UNITS, N_HIDDEN_LAYERS,
                                grad_norm=GRAD_NORM, LLH_var=1.0, state_LLH_var=0.1, noise_dim=3, initial_logvar=-40.0, device=DEVICE,
                                lr=1e-3, entropy=10.0)
#network.load(PATH)

total_steps = 0

# Train on initial dataset
dataset_state = []
dataset_action = []
dataset_rewards = []
dataset_dones = []
dataset_next_states = []
max_dataset_len = 1e8
total_training_steps = 0
current_beta = 0.0

valid_dataset_state = []
valid_dataset_action = []
valid_dataset_rewards = []
valid_dataset_dones = []
valid_dataset_next_states = []


def add_to_dataset(current_state, u, reward, done, next_state):
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global max_dataset_len
    global DEVICE

    x = torch.from_numpy(current_state.astype(np.float32)).to(DEVICE)
    a = torch.from_numpy(u.astype(np.float32)).to(DEVICE)
    r = torch.from_numpy(np.array([reward]).astype(np.float32)).to(DEVICE)
    d = torch.from_numpy(np.array([done]).astype(np.float32)).to(DEVICE)
    ns = torch.from_numpy(next_state.astype(np.float32)).to(DEVICE)

    dataset_state.append(x)
    dataset_action.append(a)
    dataset_rewards.append(r)
    dataset_dones.append(d)
    dataset_next_states.append(ns)

    if len(dataset_state) > max_dataset_len:
        dataset_state = dataset_state[1:]
        dataset_action = dataset_action[1:]
        dataset_rewards = dataset_rewards[1:]
        dataset_dones = dataset_dones[1:]
        dataset_next_states = dataset_next_states[1:]


def add_to_valid_dataset(current_state, u, reward, done, next_state):
    global valid_dataset_state
    global valid_dataset_action
    global valid_dataset_rewards
    global valid_dataset_dones
    global valid_dataset_next_states
    global DEVICE

    x = torch.from_numpy(current_state.astype(np.float32)).to(DEVICE)
    a = torch.from_numpy(u.astype(np.float32)).to(DEVICE)
    r = torch.from_numpy(np.array([reward]).astype(np.float32)).to(DEVICE)
    d = torch.from_numpy(np.array([done]).astype(np.float32)).to(DEVICE)
    ns = torch.from_numpy(next_state.astype(np.float32)).to(DEVICE)

    valid_dataset_state.append(x)
    valid_dataset_action.append(a)
    valid_dataset_rewards.append(r)
    valid_dataset_dones.append(d)
    valid_dataset_next_states.append(ns)


def train_network(network, training_epochs=1000):
    global PATH
    global DEVICE
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global total_training_steps
    global BETA_THRESHOLD
    global BETA_STEPSIZE
    global current_beta
    global DATASET_BATCH_SIZE
    global K_VALUE
    global J_VALUE
    global EPOCHS_PER_DATASET_ITR
    
    minibatch_size = min(DATASET_BATCH_SIZE, len(dataset_state))

    epoch = 0

    valid_s = torch.stack(valid_dataset_state, dim=0)
    valid_a = torch.stack(valid_dataset_action, dim=0)
    valid_r = torch.stack(valid_dataset_rewards, dim=0)
    valid_d = torch.stack(valid_dataset_dones, dim=0)
    valid_ns = torch.stack(valid_dataset_next_states, dim=0)

    for epoch in range(EPOCHS_PER_DATASET_ITR):        
    #while np.mean(neg_log_lik) > 0.1:
        train_idx = np.random.randint(0, len(dataset_state), (minibatch_size))

        mb_s = [dataset_state[idx] for idx in train_idx]
        mb_a = [dataset_action[idx] for idx in train_idx]
        mb_r = [dataset_rewards[idx] for idx in train_idx]
        mb_d = [dataset_dones[idx] for idx in train_idx]
        mb_ns = [dataset_next_states[idx] for idx in train_idx]

        mb_s = torch.stack(mb_s, dim=0)
        mb_a = torch.stack(mb_a, dim=0)
        mb_r = torch.stack(mb_r, dim=0)
        mb_d = torch.stack(mb_d, dim=0)
        mb_ns = torch.stack(mb_ns, dim=0)

        network.K = K_VALUE
        network.J = J_VALUE

        if total_training_steps > BETA_THRESHOLD:
            current_beta += BETA_STEPSIZE
        
        network.beta = min(1.0, current_beta)
        
        loss, log_lik, log_H, err = network.train(mb_s, mb_a, mb_r, mb_d, mb_ns)
            
        epoch += 1
        total_training_steps += 1
        
        print(f"Epoch = {epoch} ; Likelihood = {log_lik} ; H = {log_H} ; Err = {err}")
        sys.stdout.flush()

        #if epoch % 1000 == 0:
        #    network.save(PATH)


# Warm-up phase
print("Warming up")
print()

episode_steps = 0
#env.render()
current_state = env.reset()

while episode_steps < N_STEPS_PER_EXPLORATION:
    greedy_u = env.action_space.sample()
    #greedy_u = policy.act(current_state, sample=True)

    next_state, rew, done, _ = env.step(greedy_u)

    add_to_dataset(current_state, greedy_u, rew, done, next_state)

    episode_steps += 1

    current_state = next_state.copy()

    if done:
        #env.render()
        current_state = env.reset()

for _ in range(5000):
    greedy_u = env.action_space.sample()
    next_state, rew, done, _ = env.step(greedy_u)
    add_to_valid_dataset(current_state, greedy_u, rew, 0, next_state)
    current_state = next_state.copy()
    if done:
        #env.render()
        current_state = env.reset()
        
train_network(network, 10)
#env.render()
current_state = env.reset()
w_s, w_r, w_d = network.sample(1)
episode_steps = 0

while episode_steps < 20:
    greedy_u = env.action_space.sample()
    next_state, rew, done, _ = env.step(greedy_u)

    obs = torch.from_numpy(current_state)
    obs = obs.unsqueeze(dim=0)
    action = torch.from_numpy(greedy_u)
    action = action.unsqueeze(dim=0)
    predicted_next_state = network.primarynet_s(obs, action, w_s)
    predicted_next_state = obs + predicted_next_state[:, 0, predicted_next_state.shape[-1] // 2]

    print("Actual next state:", next_state)
    print("Predic next state:", predicted_next_state)
    print()
    sys.stdout.flush()

    episode_steps += 1

    current_state = next_state.copy()

exit()

for loop_n in range(N_OVERALL_LOOPS):
    print(f"Overall loop {loop_n}")
    print()

    #policy = iddps.sample_policy(current_state, network, dataset_state)
    policy = mdp_solver.solve(current_state, network, dataset_state)

    episode_steps = 0

    while episode_steps < N_STEPS_PER_EXPLORATION:
        greedy_u = policy.act(current_state, sample=True)

        next_state, rew, done, _ = env.step(greedy_u)

        add_to_dataset(current_state, greedy_u, rew, 0, next_state)

        current_state = next_state.copy()

        if done:
            env.render()
            current_state = env.reset()
        
        episode_steps += 1
    
    train_network(network, 10)
    #visualize_weights(network, 100, "./visual_cube_weights")
