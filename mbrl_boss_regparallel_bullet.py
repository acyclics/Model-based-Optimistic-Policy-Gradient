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

from rl.msi_mdp import Multi_SI_BayesNetwork
from rl.mdp_solver.boss_solver_regparallel import Boss_solver
from rl.logs import log_statistics


# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)
sys.stdout.flush()
nprocs = comm.Get_size()

# Overall hyperparameters
parser = argparse.ArgumentParser(description='mbrl_iddps_parallel_bullet')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=710, help='Random seed')
parser.add_argument('--env', type=str, default='AntPyBulletEnv-v0', help='Bullet env name')
parser.add_argument('--T-max', type=int, default=int(1e5), metavar='STEPS', help='Number of total environment steps (excluding evaluation steps)')
parser.add_argument('--n-explore-steps', type=int, default=int(1e3), metavar='N_E_STEPS', help='Number of steps taken per exploration')
parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of episodes for evaluating agent')
parser.add_argument('--dataset-len', type=int, default=int(1e8), help='Max length of dataset (transitions, rewards, etc)')
parser.add_argument('--terminal-done', type=bool, default=True, help='Whether end of episode should be counted as done')
parser.add_argument('--resume', type=bool, default=False, help='Whether to resume training from previous session')
# Network parameters
parser.add_argument('--noise-dim', type=int, default=3, help='Dimension of noise variable')
parser.add_argument('--llh-threshold', type=float, default=-1.0, help='Log Likelihood to reach before network training stops')
parser.add_argument('--n-last-llhs', type=int, default=50, help='Number of log-likelihoods to average')
parser.add_argument('--beta-threshold', type=int, default=0, help='Number of network training steps before increasing beta')
parser.add_argument('--beta-stepsize', type=float, default=1, help='Beta step size')
parser.add_argument('--network-batchsize', type=int, default=64, help='Network batch size')
parser.add_argument('--network-epochs', type=int, default=250, help='Epochs per network dataset iteration')
parser.add_argument('--K-value', type=int, default=1, help='Network K value')
parser.add_argument('--J-value', type=int, default=1, help='Network J value')
parser.add_argument('--network-h-units', type=int, default=128, help='Number of hidden units for network')
parser.add_argument('--network-h-layers', type=int, default=5, help='Number of hidden layers for network')
parser.add_argument('--network-grad-norm', type=float, default=5.0, help='Grad norm for network')
parser.add_argument('--network-info-itr', type=int, default=50, help='Number of epochs before printing progress')
# Model-based parameters
parser.add_argument('--plan-steps', type=int, default=20, help='Number of planning steps')
parser.add_argument('--mdp-epochs', type=int, default=1000, help='Number of mdp epochs')
parser.add_argument('--policy-epochs', type=int, default=3, help='Number of policy epochs')
parser.add_argument('--critic-epochs', type=int, default=5, help='Number of critic epochs')
parser.add_argument('--critic-batchsize', type=int, default=32, help='Critic batch size')
parser.add_argument('--replay-capacity', type=int, default=int(1e5), help='Capacity of replay buffer')
# Actor/Critic parameters
parser.add_argument('--tau', type=float, default=0.005, help='Tau for critic')
parser.add_argument('--actor-lr', type=float, default=1e-4, help='Learning rate for actor')
parser.add_argument('--critic-lr', type=float, default=1e-4, help='Learning rate for critic')
parser.add_argument('--gamma', type=float, default=0.99, help='Return decay')
parser.add_argument('--alpha', type=float, default=0.1, help='SAC alpha')
parser.add_argument('--target-update-freq', type=int, default=2, help='Target update frequency')
parser.add_argument('--actor-h-units', type=int, default=64, help='Number of hidden units for actor')
parser.add_argument('--actor-h-layers', type=int, default=3, help='Number of hidden layers for actor')
parser.add_argument('--critic-h-units', type=int, default=64, help='Number of hidden units for critic')
parser.add_argument('--critic-h-layers', type=int, default=3, help='Number of hidden layers for critic')
# BOSS parameters
parser.add_argument('--rollout-itrs', type=int, default=3, help='Number of rollouts for evaluating a policy in learned world model')
parser.add_argument('--reward-range', nargs="+", type=float, help='Min and max reward for the env: --reward-range min max')
parser.add_argument('--z-range', nargs="+", type=float, default=[2.0, 2.0, 2.0], help='Range for actor choosing z')
parser.add_argument('--obs-scale', nargs="+", type=float, help='Scaling for observations')

# Parse arguments
#if rank == 0:
#    DEVICE = 'cuda'
DEVICE = 'cuda'

args = parser.parse_args()
N_OVERALL_LOOPS = int(args.T_max / args.n_explore_steps) - 1

# Set seeds
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#random.seed(args.seed)

# Setup environment variables
env = gym.make(args.env)
eval_env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high
actor_logstd_bounds = [-5, 2]

# Setup network
NET_PATH = f"./implicit/multi_semi_implicit/model_for_{args.env}_experiment_{args.id}"
network = Multi_SI_BayesNetwork(obs_dim, action_dim, args.obs_scale,
                                args.network_h_units, args.network_h_layers,
                                grad_norm=args.network_grad_norm, LLH_var=1.0,
                                state_LLH_var=0.1,
                                noise_dim=args.noise_dim, device=DEVICE)
if args.resume:
    network.load(NET_PATH)

# Solvers
BOSS_PATH = f"./implicit/multi_semi_implicit/boss_for_{args.env}_experiment_{args.id}"
boss_solver = Boss_solver(obs_dim=obs_dim, action_dim=action_dim, horizon=args.plan_steps, epochs=args.mdp_epochs,
              actor_iterations_per_epoch=args.policy_epochs, critic_iterations_per_epoch=args.critic_epochs,
              critic_batch_size=args.critic_batchsize, action_range=action_range,
              z_range=args.z_range, noise_dim=args.noise_dim, nprocs=nprocs, comm=comm, device=DEVICE,
              actor_logstd_bounds=actor_logstd_bounds, actor_hidden_dim=args.actor_h_units, actor_hidden_layers=args.actor_h_layers,
              critic_hidden_dim=args.critic_h_units, critic_hidden_layers=args.critic_h_layers, critic_tau=args.tau,
              actor_lr=args.actor_lr, critic_lr=args.critic_lr, capacity=args.replay_capacity, gamma=args.gamma, alpha=args.alpha,
              target_update_frequency=args.target_update_freq)
if args.resume:
    boss_solver.load(BOSS_PATH)

# Directories and metrics
metrics = {'steps': [], 'rewards': []}
results_dir = os.path.join('results', args.id, args.env)
agent_dir = os.path.join('agents', args.id, args.env)
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(agent_dir).mkdir(parents=True, exist_ok=True)

# Train on initial dataset
cpu_dataset_state = []
dataset_state = []
dataset_action = []
dataset_rewards = []
dataset_dones = []
dataset_next_states = []
total_training_steps = 0
current_beta = 0.0


def add_to_dataset(current_state, u, reward, done, next_state):
    global args
    global cpu_dataset_state
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global DEVICE

    if not args.terminal_done:
        """
            Some environments are "done" when episode length is reached.
            This is a quick hack for environments that only "done"
            when episode length is reached.
        """
        done = False

    x = torch.from_numpy(current_state.astype(np.float32)).to(DEVICE)
    a = torch.from_numpy(u.astype(np.float32)).to(DEVICE)
    r = torch.from_numpy(np.array([reward]).astype(np.float32)).to(DEVICE)
    d = torch.from_numpy(np.array([done]).astype(np.float32)).to(DEVICE)
    ns = torch.from_numpy(next_state.astype(np.float32)).to(DEVICE)

    #cpu_dataset_state.append(current_state.astype(np.float32))
    dataset_state.append(x)
    dataset_action.append(a)
    dataset_rewards.append(r)
    dataset_dones.append(d)
    dataset_next_states.append(ns)

    cpu_dataset_state.append(x.clone())

    if len(dataset_state) > args.dataset_len:
        dataset_state = dataset_state[1:]
        dataset_action = dataset_action[1:]
        dataset_rewards = dataset_rewards[1:]
        dataset_dones = dataset_dones[1:]
        dataset_next_states = dataset_next_states[1:]


def train_network(network, training_epochs=1000):
    global args
    global NET_PATH
    global DEVICE
    global dataset_state
    global dataset_action
    global dataset_rewards
    global dataset_dones
    global dataset_next_states
    global total_training_steps
    global current_beta
    
    minibatch_size = min(args.network_batchsize, len(dataset_state))
    avg_llh = -100.0
    n_last_llhs = args.n_last_llhs
    past_llhs = [-100.0 for _ in range(n_last_llhs)]
    pe_idx = 0
    epoch = 0

    while avg_llh < args.llh_threshold or epoch < args.network_epochs:
    #for epoch in range(1):        
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

        network.K = args.K_value
        network.J = args.J_value

        if total_training_steps > args.beta_threshold:
            current_beta += args.beta_stepsize
        
        network.beta = min(1.0, current_beta)
        
        loss, log_lik, log_H, err = network.train(mb_s, mb_a, mb_r, mb_d, mb_ns)

        past_llhs[pe_idx] = log_lik
        pe_idx = (pe_idx + 1) % n_last_llhs
        avg_llh = 0
        for past_llh in past_llhs:
            avg_llh += past_llh
        avg_llh /= n_last_llhs
        
        total_training_steps += 1
        epoch += 1

        if epoch % args.network_info_itr == 0 or epoch + 1 == args.network_epochs:
            print(f"Epoch = {epoch} ; Average loss = {loss} ; Likelihood = {log_lik} ; H = {log_H} ; Err = {err} ; Beta = {network.beta}")
            sys.stdout.flush()

    network.save(NET_PATH)


if rank == 0:
    # Warm-up phase
    total_steps_taken = 0

    print("Warming up")
    print()
    sys.stdout.flush()

    episode_steps = 0
    current_state = env.reset()

    while episode_steps < args.n_explore_steps:
        greedy_u = env.action_space.sample()

        next_state, rew, done, _ = env.step(greedy_u)

        add_to_dataset(current_state, greedy_u, rew, done, next_state)

        episode_steps += 1
        total_steps_taken += 1
        
        current_state = next_state.copy()

        if done:
            current_state = env.reset()
    
    if not args.resume:
        train_network(network, 10)

    for loop_n in range(N_OVERALL_LOOPS):
        current_state = env.reset()

        #network_state_dict = network.get_state_dict(cpu=True)
        network_state_dict = network.get_state_dict()

        package = {
            'network_state_dict': network_state_dict,
            'dataset_to_send': cpu_dataset_state
        }

        for p in range(nprocs - 1):
            comm.send(package, dest=p+1)
        
        cpu_dataset_state = []
        
        policy, critic = boss_solver.solve(network, dataset_state, verbose=True)
        boss_solver.save(BOSS_PATH)
        policy = boss_solver.make_policy(policy)

        done = False
        episode_steps = 0
        current_state = env.reset()
        #for loop_rollout in range(args.rollout_itrs):
        while episode_steps < args.n_explore_steps:
            #done = False
            #current_state = env.reset()        
            #while not done:
            greedy_u = policy.act(current_state, sample=True)
            next_state, rew, done, _ = env.step(greedy_u)
            episode_steps += 1
            total_steps_taken += 1

            add_to_dataset(current_state, greedy_u, rew, done, next_state)
            current_state = next_state.copy()

            if done:
                current_state = env.reset()

        policy.save(os.path.join(agent_dir, f"agent_loop{loop_n}"))
        log_statistics(total_steps_taken, eval_env, policy, metrics, results_dir, args.n_eval_episodes)
        train_network(network, 10)
else:
    for _ in range(N_OVERALL_LOOPS):
        package = comm.recv(source=0)
        print(f"Rank {rank} received package")
        sys.stdout.flush()

        network_state_dict = package['network_state_dict']
        network.load_state_dict(network_state_dict)
        dataset_to_recv = package['dataset_to_send']
        dataset_state.extend(dataset_to_recv)

        boss_solver.reset()
        boss_solver.process_reset()

        while True:
            package = comm.recv(source=0)

            if not package['continue']:
                break
            
            collect_data = package['collect_data']

            if collect_data:
                return_package = boss_solver.process_fill_replay(package['policy'], package['critic'], network, dataset_state)
            else:
                boss_solver.log_alpha.data = package['logalpha']
                policy_grads, actor_loss, total_log_probs, transitions = boss_solver.collect_rollouts(package['policy'], package['critic'], network, dataset_state)
                return_package = {
                    'policy_grads': policy_grads,
                    'actor_loss': actor_loss,
                    'total_log_probs': total_log_probs,
                    'transitions': transitions
                }
            
            comm.send(return_package, dest=0)
        
        print(f"Rank {rank} ending iteration")
        sys.stdout.flush()

env.close()
