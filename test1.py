import sys
import os
import numpy as np
import random
import torch
from pathlib import Path
import argparse
from copy import deepcopy

from mbopg.bh_mdp import Multi_SI_BayesNetwork
from mbopg.mbrl import MBRL_solver
from logs import log_rewards, log_likelihood
from toyenv import ToyEnv


# Overall hyperparameters
parser = argparse.ArgumentParser(description='mbopg_parallel')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=710, help='Random seed')
parser.add_argument('--env', type=str, default='AntPyBulletEnv-v0', help='Environment name')
parser.add_argument('--T-max', type=int, default=int(1e5), metavar='STEPS', help='Number of total environment steps (excluding evaluation steps)')
parser.add_argument('--n-explore-steps', type=int, default=int(1e3), metavar='N_E_STEPS', help='Number of steps taken per exploration')
parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of episodes for evaluating agent')
parser.add_argument('--dataset-len', type=int, default=int(1e8), help='Max length of dataset (transitions, rewards, etc)')
parser.add_argument('--terminal-done', type=bool, default=True, help='Whether end of episode should be counted as done')
parser.add_argument('--resume', type=bool, default=False, help='Whether to resume training from previous session')

# Network parameters
parser.add_argument('--llh-threshold', type=float, default=-1.0, help='Log Likelihood to reach before network training stops')
parser.add_argument('--beta-threshold', type=int, default=0, help='Number of network training steps before increasing beta')
parser.add_argument('--entropy', type=float, default=1.0, help='Entropy of Network')
parser.add_argument('--beta-stepsize', type=float, default=1, help='Beta step size')
parser.add_argument('--network-batchsize', type=int, default=64, help='Network batch size')
parser.add_argument('--network-epochs', type=int, default=250, help='Epochs per network dataset iteration')
parser.add_argument('--K-value', type=int, default=1, help='Network K value')
parser.add_argument('--J-value', type=int, default=1, help='Network J value')
parser.add_argument('--network-h-units', type=int, default=128, help='Number of hidden units for network')
parser.add_argument('--network-h-layers', type=int, default=5, help='Number of hidden layers for network')
parser.add_argument('--network-grad-norm', type=float, default=5.0, help='Grad norm for network')
parser.add_argument('--network-info-itr', type=int, default=50, help='Number of epochs before printing progress')
parser.add_argument('--network-lr', type=float, default=1e-4, help='LR for network')
parser.add_argument('--network-w-units', type=int, default=128, help='Number of weight units for network')

# Model-based parameters
parser.add_argument('--plan-steps', type=int, default=10, help='Number of planning steps')
parser.add_argument('--mdp-epochs', type=int, default=1000, help='Number of mdp epochs')
parser.add_argument('--policy-epochs', type=int, default=3, help='Number of policy epochs')
parser.add_argument('--replay-capacity', type=int, default=int(1e5), help='Capacity of replay buffer')

# Actor/Critic parameters
parser.add_argument('--tau', type=float, default=0.005, help='Tau for critic')
parser.add_argument('--actor-lr', type=float, default=1e-4, help='Learning rate for actor')
parser.add_argument('--critic-lr', type=float, default=1e-4, help='Learning rate for critic')
parser.add_argument('--gamma', type=float, default=0.99, help='Return decay')
parser.add_argument('--alpha', type=float, default=0.1, help='Entropy alpha')
parser.add_argument('--noise-alpha', type=float, default=0.1, help='Entropy noise alpha')
parser.add_argument('--target-update-frequency', type=int, default=2, help='Target update frequency')
parser.add_argument('--actor-h-units', type=int, default=64, help='Number of hidden units for actor')
parser.add_argument('--actor-h-layers', type=int, default=3, help='Number of hidden layers for actor')
parser.add_argument('--critic-h-units', type=int, default=64, help='Number of hidden units for critic')
parser.add_argument('--critic-h-layers', type=int, default=3, help='Number of hidden layers for critic')

# MBOPG parameters
parser.add_argument('--obs-scale', type=float, help='Scaling for observations')
noise_dim = 3
z_range = [2.0, 2.0, 2.0]
actor_logstd_bounds = [-2, 2]

# Finalize
DEVICE = 'cuda'
args = parser.parse_args()

# Directories and metrics
metrics = {'steps_rewards': [], 'rewards': [], 'steps_likelihood': [], 'likelihood': []}
base_dir = os.path.join(args.id, args.env, f"seed_{args.seed}")
results_dir = os.path.join(base_dir, 'results')
agents_dir = os.path.join(base_dir, 'agents')
net_path = os.path.join(base_dir, 'hypernetwork_model')
mbrl_path = os.path.join(base_dir, 'mbrl_model')
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(agents_dir).mkdir(parents=True, exist_ok=True)

# Set seeds
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#random.seed(args.seed)

# Setup environment variables
env = ToyEnv()
obs_dim = env.obs_dim
action_dim = env.act_dim
action_range = env.action_range

# Setup network
state_min = np.ones((obs_dim), dtype=np.float32) * 1000000.0
state_max = np.ones((obs_dim), dtype=np.float32) * -1000000.0
current_beta = 0.0
total_training_steps = 0

network = Multi_SI_BayesNetwork(obs_dim, action_dim, args.obs_scale, state_min, state_max,
                            args.network_h_units, args.network_h_layers,
                            grad_norm=args.network_grad_norm, LLH_var=0.1,
                            state_LLH_var=1.0, initial_logvar=np.log(0.1),
                            noise_dim=noise_dim, entropy=args.entropy, device=DEVICE)
if args.resume:
    network.load(net_path)

# Solvers
mbrl_solver = MBRL_solver(obs_dim=obs_dim, action_dim=action_dim, horizon=args.plan_steps, epochs=args.mdp_epochs,
              actor_iterations_per_epoch=args.policy_epochs,
              target_update_frequency=args.target_update_frequency,
              action_range=action_range,
              z_range=z_range, noise_dim=noise_dim, device=DEVICE,
              actor_logstd_bounds=actor_logstd_bounds, actor_hidden_dim=args.actor_h_units, actor_hidden_layers=args.actor_h_layers,
              critic_hidden_dim=args.critic_h_units, critic_hidden_layers=args.critic_h_layers, tau=args.tau,
              actor_lr=args.actor_lr, critic_lr=args.critic_lr, capacity=args.replay_capacity, gamma=args.gamma, alpha=args.alpha,
              noise_alpha=args.noise_alpha)

# Train on initial dataset
dataset_state = []
dataset_action = []
dataset_rewards = []
dataset_dones = []
dataset_next_states = []
dataset_logprob = []


def add_to_dataset():
    global args
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global dataset_logprob
    global state_min
    global state_max
    global DEVICE
    global mbrl_solver
    global env

    ds, da, dr, dns, dd = env.initial_dataset()

    for idx in range(len(ds)):
        current_state = ds[idx]
        u = da[idx]
        reward = dr[idx]
        done = dd[idx]
        next_state = dns[idx]
        x = torch.from_numpy(current_state.astype(np.float32)).to(DEVICE)
        a = torch.from_numpy(u.astype(np.float32)).to(DEVICE)
        r = torch.from_numpy(np.array([reward]).astype(np.float32)).to(DEVICE)
        d = torch.from_numpy(np.array([done]).astype(np.float32)).to(DEVICE)
        ns = torch.from_numpy(next_state.astype(np.float32)).to(DEVICE)

        state_min = np.minimum(state_min, current_state.astype(np.float32))
        state_max = np.maximum(state_max, current_state.astype(np.float32))

        dataset_state.append(x)
        dataset_action.append(a)
        dataset_rewards.append(r)
        dataset_dones.append(d)
        dataset_next_states.append(ns)
    
    return ds, da, dr, dns, dd


ds, da, dr, dns, dd = add_to_dataset()

max_r = -np.inf
min_r = np.inf

for r in dr:
    max_r = max(max_r, r)
    min_r = min(min_r, r)

print(f"Max reward = {max_r} ; Min reward = {min_r}")
sys.stdout.flush()


def train_network(n_itrs):
    global network
    global args
    global net_path
    global DEVICE
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global state_min
    global state_max
    global current_beta

    network.set_obs_trans(state_min, state_max)

    total_training_steps = 0
    minibatch_size = min(args.network_batchsize, len(dataset_state))
    epoch = 0
    avg_llh = 0
    initial_llh = None

    dataset = list(zip(dataset_state, dataset_action, dataset_rewards, dataset_dones, dataset_next_states))

    for _ in range(n_itrs):
        avg_llh = 0
        avg_loss = 0
        avg_H = 0
        avg_err = 0

        random.shuffle(dataset)
        dataset_state, dataset_action, dataset_rewards, dataset_dones, dataset_next_states = zip(*dataset)

        for idx in range(0, len(dataset_state), minibatch_size):
            mb_s = dataset_state[idx : idx + minibatch_size]
            mb_a = dataset_action[idx : idx + minibatch_size]
            mb_r = dataset_rewards[idx : idx + minibatch_size]
            mb_d = dataset_dones[idx : idx + minibatch_size]
            mb_ns = dataset_next_states[idx : idx + minibatch_size]

            mb_s = torch.stack(mb_s, dim=0)
            mb_a = torch.stack(mb_a, dim=0)
            mb_r = torch.stack(mb_r, dim=0)
            mb_d = torch.stack(mb_d, dim=0)
            mb_ns = torch.stack(mb_ns, dim=0)

            network.K = args.K_value
            network.J = args.J_value

            if total_training_steps > args.beta_threshold:
                current_beta += args.beta_stepsize
            
            network.beta = min(1.0, current_beta)
            
            loss, log_lik, log_H, err = network.train(mb_s, mb_a, mb_r, mb_d, mb_ns)

            avg_llh += log_lik
            avg_loss += loss
            avg_H += log_H
            avg_err += err
            
            total_training_steps += 1
        
        epoch += 1

        avg_llh /= (len(dataset_state) / minibatch_size)
        avg_loss /= (len(dataset_state) / minibatch_size)
        avg_H /= (len(dataset_state) / minibatch_size)
        avg_err /= (len(dataset_state) / minibatch_size)

        if initial_llh == None:
            initial_llh = avg_llh

        print(f"Epoch = {epoch} ; Average loss = {avg_loss} ; Likelihood = {avg_llh} ; H = {avg_H} ; Err = {avg_err} ; Beta = {network.beta}")
        sys.stdout.flush()

    dataset_state = list(dataset_state)
    dataset_action = list(dataset_action)
    dataset_rewards = list(dataset_rewards)
    dataset_dones = list(dataset_dones)
    dataset_next_states = list(dataset_next_states)

    #network.save(net_path)

    return initial_llh


total_steps_taken = 0
initial_network_epochs = args.network_epochs

print("Warming up")
print()
sys.stdout.flush()

if True:
    print("Training network")
    sys.stdout.flush()
    train_network(3000)
else:
    print(state_min, state_max)
    network.load(net_path)
    network.set_obs_trans(state_min, state_max)

all_prev_rews = []
all_new_rews = []

for idx in range(0, 1):
    rrr = 0.0

    for _ in range(10):
        obs = ds[idx]
        print(obs)
        sys.stdout.flush()
        prev_action = da[idx]
        #prev_action = np.random.uniform(-1.0, 1.0, (3)) * 0.01
        prev_reward = dr[idx]
        obs = torch.from_numpy(obs).unsqueeze(dim=0).to(DEVICE).float()
        prev_action = torch.from_numpy(prev_action).unsqueeze(dim=0).to(DEVICE).float()

        z = (-2.0 - 2.0) * torch.rand((1, noise_dim)) + 2.0
        z = z.float().to(DEVICE)
        z.requires_grad=True

        for i in range(1000):
            w = network.sample(1, z)
            next_obs, reward, done = network.primarynet(obs, prev_action, w)
            if i == 0:
                initial_rew = reward
            z_grad = torch.autograd.grad(torch.mean(reward), z)
            #z = torch.add(z, )
            z = z + z_grad[0] * 0.1
            z = torch.clamp(z, -2.0, 2.0)

        rrr += abs(reward - initial_rew)
        
    rrr = rrr / 10
    all_prev_rews.append(rrr)

print(all_prev_rews)
new_r = 0.0

for _ in range(10):
    idx = -1

    obs = ds[idx]
    obs = np.random.uniform(0.5, 0.6, (3)).astype(np.float32)
    print(obs)
    sys.stdout.flush()
    prev_action = da[idx]
    #prev_action = np.random.uniform(-1.0, 1.0, (3)) * 0.01
    prev_reward = dr[idx]
    obs = torch.from_numpy(obs).unsqueeze(dim=0).to(DEVICE).float()
    prev_action = torch.from_numpy(prev_action).unsqueeze(dim=0).to(DEVICE).float()

    z = (-2.0 - 2.0) * torch.rand((1, noise_dim)) + 2.0
    z = z.float().to(DEVICE)
    z.requires_grad=True

    for i in range(1000):
        w = network.sample(1, z)
        next_obs, reward, done = network.primarynet(obs, prev_action, w)
        if i == 0:
            initial_rew = reward
        z_grad = torch.autograd.grad(torch.mean(reward), z)
        #z = torch.add(z, )
        z = z + z_grad[0] * 0.1
        z = torch.clamp(z, -2.0, 2.0)
    
    new_r += abs(reward - initial_rew)

print(new_r / 10)
