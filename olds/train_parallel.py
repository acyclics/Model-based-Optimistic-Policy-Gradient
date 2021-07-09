import sys
import os
import numpy as np
import random
import torch
from pathlib import Path
import argparse
from mpi4py import MPI
from copy import deepcopy

import gym
import pybulletgym

from mbopg.bh_mdp import Multi_SI_BayesNetwork
from mbopg.mbrl_parallel import MBRL_solver
from logs import log_statistics


# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)
sys.stdout.flush()
nprocs = comm.Get_size()

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
parser.add_argument('--policy-repeats', type=int, default=3, help='Number of policy repeats')
parser.add_argument('--replay-capacity', type=int, default=int(1e5), help='Capacity of replay buffer')

# Actor/Critic parameters
parser.add_argument('--tau', type=float, default=0.005, help='Tau for critic')
parser.add_argument('--actor-lr', type=float, default=1e-4, help='Learning rate for actor')
parser.add_argument('--critic-lr', type=float, default=1e-4, help='Learning rate for critic')
parser.add_argument('--gamma', type=float, default=0.99, help='Return decay')
parser.add_argument('--alpha', type=float, default=0.1, help='Entropy alpha')
parser.add_argument('--actor-h-units', type=int, default=64, help='Number of hidden units for actor')
parser.add_argument('--actor-h-layers', type=int, default=3, help='Number of hidden layers for actor')
parser.add_argument('--critic-h-units', type=int, default=64, help='Number of hidden units for critic')
parser.add_argument('--critic-h-layers', type=int, default=3, help='Number of hidden layers for critic')
parser.add_argument('--action-noise', type=float, default=0.4, help='Action noise')
parser.add_argument('--target-noise', type=float, default=0.1, help='Target noise')
parser.add_argument('--noise-clip', type=float, default=0.2, help='Noise clip')
parser.add_argument('--multi-steps', type=int, default=3, help='Multi steps')

# MBOPG parameters
parser.add_argument('--obs-scale', type=float, help='Scaling for observations')
parser.add_argument('--surrogate-epochs', type=int, default=int(1e5), help='Number of surrogate epochs')
parser.add_argument('--surrogate-batchsize', type=int, default=200, help='Number of surrogate batch size')
parser.add_argument('--surrogate-tau', type=float, default=0.005, help='Tau for surrogate critic')
parser.add_argument('--surrogate-target-update-frequency', type=int, default=2, help='Target update frequency')
parser.add_argument('--clear-steps', type=int, default=5, help='Clear steps')
noise_dim = 3
z_range = [2.0, 2.0, 2.0]
actor_logstd_bounds = [-5, 2]

# Finalize
if rank == 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cuda'
args = parser.parse_args()
N_OVERALL_LOOPS = int(args.T_max / args.n_explore_steps) - 1

# Directories and metrics
metrics = {'steps': [], 'rewards': []}
base_dir = os.path.join(args.id, args.env, f"seed_{args.seed}")
results_dir = os.path.join(base_dir, 'results')
agents_dir = os.path.join(base_dir, 'agents')
net_path = os.path.join(base_dir, 'hypernetwork_model')
mbrl_path = os.path.join(base_dir, 'mbrl_model')
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(agents_dir).mkdir(parents=True, exist_ok=True)

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# Setup environment variables
env = gym.make(args.env)
eval_env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high

# Setup network
network = None
#if args.resume:
#    network.load(net_path)

# Solvers
mbrl_solver = MBRL_solver(obs_dim=obs_dim, action_dim=action_dim, horizon=args.plan_steps, epochs=args.mdp_epochs,
              comm=comm,
              actor_iterations_per_epoch=args.policy_epochs, actor_repeat_per_epoch=args.policy_repeats,
              surrogate_epochs=args.surrogate_epochs,
              surrogate_batchsize=args.surrogate_batchsize,
              surrogate_tau=args.surrogate_tau,
              surrogate_target_update_frequency=args.surrogate_target_update_frequency,
              action_range=action_range,
              z_range=z_range, noise_dim=noise_dim, device=DEVICE,
              actor_logstd_bounds=actor_logstd_bounds, actor_hidden_dim=args.actor_h_units, actor_hidden_layers=args.actor_h_layers,
              critic_hidden_dim=args.critic_h_units, critic_hidden_layers=args.critic_h_layers, tau=args.tau,
              actor_lr=args.actor_lr, critic_lr=args.critic_lr, capacity=args.replay_capacity, gamma=args.gamma, alpha=args.alpha,
              action_noise=args.action_noise,
              target_noise=args.target_noise, noise_clip=args.noise_clip, multi_steps=args.multi_steps)
#if args.resume:
#    mbrl_solver.load(mbrl_path)

# Train on initial dataset
cpu_dataset_state = []
cpu_dataset_action = []
cpu_dataset_reward = []
cpu_dataset_next_state = []
cpu_dataset_done = []
cpu_dataset_done_no_max = []

dataset_state = []
dataset_action = []
dataset_rewards = []
dataset_dones = []
dataset_next_states = []

state_min = np.ones((obs_dim), dtype=np.float32) * 1000000.0
state_max = np.ones((obs_dim), dtype=np.float32) * -1000000.0


def add_to_dataset(current_state, u, reward, done, next_state, current_step):
    global args
    global cpu_dataset_state
    global cpu_dataset_action
    global cpu_dataset_reward
    global cpu_dataset_next_state
    global cpu_dataset_done
    global cpu_dataset_done_no_max
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global state_min
    global state_max
    global DEVICE
    global mbrl_solver
    global env

    done_no_max = done

    if not args.terminal_done:
        """
            Some environments are "done" when episode length is reached.
            This is a quick hack for environments that only "done"
            when episode length is reached.
        """
        if env._max_episode_steps == current_step - 1:
            done = False

    cpu_dataset_state.append(current_state.astype(np.float32))
    cpu_dataset_action.append(u.astype(np.float32))
    cpu_dataset_reward.append(np.array([reward]).astype(np.float32))
    cpu_dataset_next_state.append(next_state.astype(np.float32))
    cpu_dataset_done.append(np.array([done]).astype(np.float32))
    cpu_dataset_done_no_max.append(np.array([done_no_max]).astype(np.float32))

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

    """
    mbrl_solver.surrogate_replay_buffer.add(x.unsqueeze(dim=0),
                                            a.unsqueeze(dim=0),
                                            r.unsqueeze(dim=0),
                                            ns.unsqueeze(dim=0),
                                            d.unsqueeze(dim=0),
                                            False)
    """
    mbrl_solver.surrogate_replay_buffer.add(current_state.astype(np.float32),
                                            u.astype(np.float32),
                                            np.array([reward]).astype(np.float32),
                                            next_state.astype(np.float32),
                                            np.array([done]).astype(np.float32),
                                            done_no_max)

    if len(dataset_state) > args.dataset_len:
        dataset_state = dataset_state[1:]
        dataset_action = dataset_action[1:]
        dataset_rewards = dataset_rewards[1:]
        dataset_dones = dataset_dones[1:]
        dataset_next_states = dataset_next_states[1:]


def train_network():
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

    network = Multi_SI_BayesNetwork(obs_dim, action_dim, args.obs_scale, state_min, state_max,
                            args.network_h_units, args.network_h_layers,
                            grad_norm=args.network_grad_norm, LLH_var=1.0,
                            state_LLH_var=0.1, initial_logvar=-40.0,
                            noise_dim=noise_dim, entropy=args.entropy, device=DEVICE)

    current_beta = 0.0
    total_training_steps = 0
    
    minibatch_size = min(args.network_batchsize, len(dataset_state))
    epoch = 0
    avg_llh = 0

    dataset = list(zip(dataset_state, dataset_action, dataset_rewards, dataset_dones, dataset_next_states))

    while avg_llh < args.llh_threshold or epoch < args.network_epochs:
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

        print(f"Epoch = {epoch} ; Average loss = {avg_loss} ; Likelihood = {avg_llh} ; H = {avg_H} ; Err = {avg_err} ; Beta = {network.beta}")
        sys.stdout.flush()

    dataset_state = list(dataset_state)
    dataset_action = list(dataset_action)
    dataset_rewards = list(dataset_rewards)
    dataset_dones = list(dataset_dones)
    dataset_next_states = list(dataset_next_states)

    network.save(net_path)


if rank == 0:
    total_steps_taken = 0
    initial_network_epochs = args.network_epochs

    print("Warming up")
    print()
    sys.stdout.flush()

    episode_steps = 0
    current_steps = 0
    current_state = env.reset()

    while episode_steps < args.n_explore_steps:
        greedy_u = env.action_space.sample()

        next_state, rew, done, _ = env.step(greedy_u)

        current_steps += 1
        episode_steps += 1
        total_steps_taken += 1

        add_to_dataset(current_state, greedy_u, rew, done, next_state, current_steps)

        current_state = next_state.copy()

        if done:
            current_steps = 0
            current_state = env.reset()

    if not args.resume:
        train_network()

    for loop_n in range(N_OVERALL_LOOPS):
        """
        if (loop_n + 1) % args.clear_steps == 0:
            policy, critic = mbrl_solver.clear_solve(network, dataset_state, verbose=True)
        else:
            policy, critic = mbrl_solver.solve(network, dataset_state, verbose=True)
        """
        datasets = {
            'state': cpu_dataset_state,
            'action': cpu_dataset_action,
            'reward': cpu_dataset_reward,
            'next_state': cpu_dataset_next_state,
            'done': cpu_dataset_done,
            'done_no_max': cpu_dataset_done_no_max
        }
        comm.send(datasets, dest=1)
        cpu_dataset_state = []
        cpu_dataset_action = []
        cpu_dataset_reward = []
        cpu_dataset_next_state = []
        cpu_dataset_done = []
        cpu_dataset_done_no_max = []
        mbrl_solver.solve(network, dataset_state, verbose=True)
        outputs = comm.recv()
        policy, critic = outputs['policy'], outputs['critic']

        #mbrl_solver.save(mbrl_path)
        policy = mbrl_solver.make_policy(policy)

        done = False
        episode_steps = 0
        current_steps = 0
        current_state = env.reset()

        while episode_steps < args.n_explore_steps:
            greedy_u = policy.act(current_state, sample=True)

            next_state, rew, done, _ = env.step(greedy_u)

            current_steps += 1
            episode_steps += 1
            total_steps_taken += 1

            add_to_dataset(current_state, greedy_u, rew, done, next_state, current_steps)
            current_state = next_state.copy()

            if done:
                current_steps = 0
                current_state = env.reset()

        policy.save(os.path.join(agents_dir, f"agent_loop{loop_n}"))
        log_statistics(total_steps_taken, eval_env, policy, metrics, results_dir, args.n_eval_episodes)

        args.network_epochs = min(initial_network_epochs + initial_network_epochs * (loop_n + 1), 10000)
        train_network()
else:
    for _ in range(N_OVERALL_LOOPS): 
        datasets = comm.recv()       
        for idx in range(len(datasets['state'])):   
            current_state = datasets['state'][idx]
            u = datasets['action'][idx]
            reward = datasets['reward'][idx]
            next_state = datasets['next_state'][idx]
            done = datasets['done'][idx] 
            done_no_max = datasets['done_no_max'][idx] 
            mbrl_solver.surrogate_replay_buffer.add(current_state,
                                                    u,
                                                    reward,
                                                    next_state,
                                                    done,
                                                    done_no_max)
        policy, critic = mbrl_solver.solve_surrogate()
        outputs = {
            'policy': policy,
            'critic': critic
        }
        comm.send(outputs, dest=0)
