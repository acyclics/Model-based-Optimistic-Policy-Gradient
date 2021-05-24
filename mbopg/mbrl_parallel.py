import os
import sys
import numpy as np
import torch
import math
import random
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from copy import deepcopy
from time import time

import mbopg.utils as utils
from mbopg.actor import DiagGaussianActor
from mbopg.critic import DoubleQCritic
from mbopg.artificial_replay import ArtificialReplayBuffer


class Policy():

    def __init__(self, actor, action_range, noise_dim, device):
        self.actor = actor
        self.action_range = action_range
        self.noise_dim = noise_dim
        self.device = device
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        if sample:
            action = dist.sample()
        else:
            action = dist.mean
        action = action * self.action_range
        action = action[0, 0:-self.noise_dim]
        return action.detach().cpu().numpy()
    
    def save(self, filepath):
        save_dict = {
            'actor': self.actor,
            'action_range': self.action_range,
            'noise_dim': self.noise_dim
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath, cpu=False):
        if cpu:
            save_dict = torch.load(filepath, map_location='cpu')
        else:
            save_dict = torch.load(filepath)
        self.actor = save_dict['actor']
        self.action_range = save_dict['action_range']
        self.noise_dim = save_dict['noise_dim']


class MBRL_solver(nn.Module):

    def __init__(self, obs_dim, action_dim, horizon, epochs, actor_iterations_per_epoch, critic_min_iterations_per_epoch,
                 critic_max_iterations_per_epoch, critic_loss_threshold,
                 critic_batch_size, action_range, z_range, noise_dim, nprocs, comm, device,
                 actor_logstd_bounds=[-5, 5], actor_hidden_dim=64, actor_hidden_layers=3, critic_hidden_dim=64, 
                 critic_hidden_layers=3, tau=0.005, actor_lr=1e-4, critic_lr=1e-4, actor_betas=[0.9, 0.999], 
                 critic_betas=[0.9, 0.999], capacity=1e5,
                 gamma=0.99, alpha=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim + noise_dim
        self.horizon = horizon
        self.epochs = epochs
        self.nprocs = nprocs
        self.comm = comm

        self.actor_logstd_bounds = actor_logstd_bounds
        self.actor_iterations_per_epoch = actor_iterations_per_epoch
        self.critic_min_iterations_per_epoch = critic_min_iterations_per_epoch
        self.critic_max_iterations_per_epoch = critic_max_iterations_per_epoch
        self.critic_loss_threshold = critic_loss_threshold
        self.actor_hidden_dim = actor_hidden_dim
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_hidden_layers = critic_hidden_layers
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_betas = actor_betas
        self.critic_betas = critic_betas
        self.capacity = capacity
        self.noise_dim = noise_dim

        self.gamma = gamma
        self.critic_batch_size = critic_batch_size

        action_range = list(action_range) + list(z_range)

        if device == 'cuda':
            self.action_range = torch.cuda.FloatTensor(action_range)
        else:
            self.action_range = torch.FloatTensor(action_range)
            
        self.action_range.requires_grad = False

        self.device = device

        self.actor = DiagGaussianActor(self.obs_dim, self.action_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        self.target_actor = DiagGaussianActor(self.obs_dim, self.action_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)

        self.critic = DoubleQCritic(self.obs_dim, self.action_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target = DoubleQCritic(self.obs_dim, self.action_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)

        self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=[0.9, 0.999])

        self.reset()
        
    def solve(self, network, dataset_states, verbose=True):
        self.reset()
        self._solve(network, dataset_states, verbose=verbose)
        policy = self.actor.state_dict()
        critic = self.critic.state_dict()
        return policy, critic
    
    def make_policy(self, policy):
        actor = DiagGaussianActor(self.obs_dim, self.action_dim, self.actor_hidden_dim, self.actor_hidden_layers, 
                                  self.actor_logstd_bounds).to(self.device)
        actor.load_state_dict(policy)
        policy = Policy(actor, self.action_range, self.noise_dim, self.device)
        return policy
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_critic(self):
        obs, action, reward, next_obs, not_done, _ = self.replay_buffer.sample(self.critic_batch_size)

        dist = self.actor_target(next_obs)

        next_action = dist.sample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 50.0)
        self.critic_optimizer.step()

        return critic_loss.detach()

    def update_alpha(self, total_log_probs):
        self.log_alpha_optimizer.zero_grad()
        total_log_probs = total_log_probs / (self.nprocs * self.horizon * self.actor_iterations_per_epoch)
        alpha_loss = (self.alpha * (-total_log_probs - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 5.0)
        self.log_alpha_optimizer.step()

    def _solve_once(self, network, dataset_states, is_main=False):
        actor_values = []
        total_log_probs = []

        all_obses = []
        all_actions = []
        all_rewards = []
        all_next_obses = []
        all_dones = []

        for n_itr in range(self.actor_iterations_per_epoch):
            initial_obs = dataset_states[np.random.randint(0, len(dataset_states))]
            
            if type(initial_obs).__module__ == np.__name__:
                obs = torch.from_numpy(initial_obs)
            else:
                obs = initial_obs.clone()

            obs = obs.unsqueeze(dim=0)
            
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            Ws = []
        
            for t in range(self.horizon):
                dist = self.actor(obs)
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(-1)
                out_action = action * self.action_range

                act_action = out_action[:, 0:-self.noise_dim]

                z = out_action[:, -self.noise_dim:]
                w = network.sample(1, z)
                
                next_obs, reward, done = network.primarynet(obs, act_action, w)

                next_obs = next_obs[:, 0]
                reward = reward[0, 0]
                done = done[0, 0, 0]
                
                self.replay_buffer.add(obs.detach(), action.detach(), reward.detach(), next_obs.detach(), done > 0.5, False)
                                
                states.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                total_log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                Ws.append(w)

                all_obses.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_obses.append(next_obs)
                all_dones.append(done)

                obs = next_obs

                if t + 1 == self.horizon:
                    next_state = next_obs
                    dist = self.actor(next_state)
                    next_action = dist.rsample()
                    next_log_prob = dist.log_prob(next_action).sum(-1)
                    break

            states.append(next_state)
            actions.append(next_action)
            log_probs.append(next_log_prob)
            #all_vs = []

            """
            for t1 in reversed(range(0, len(states))):
                Q1, Q2 = self.critic(states[t1], actions[t1])
                V = torch.min(Q1, Q2)
                V = V[0]

                for t2 in reversed(range(0, t1)):
                    V = rewards[t2] - self.alpha.detach() * log_probs[t2] + (1.0 - dones[t2]) * self.gamma * V

                all_vs.append(V)
            """
            
            Q1, Q2 = self.critic(states[-1], actions[-1])
            V = torch.min(Q1, Q2)
            V = V[0] - self.alpha.detach() * log_probs[-1]

            for t in reversed(range(len(rewards))):
                V = rewards[t] - self.alpha.detach() * log_probs[t] + (1.0 - dones[t]) * self.gamma * V
            
            #all_vs.append(V)
            """
            actor_Q1, actor_Q2 = self.critic(states[0], actions[0])
            actor_Q = torch.min(actor_Q1, actor_Q2)
            Q = actor_Q[0] - self.alpha.detach() * log_probs[0]

            #all_vs.append(Q)
            if Q > V:
                all_vs = Q
            else:
                all_vs = V
            """
            
            
            #all_vs = torch.stack(all_vs, dim=0)
            #all_vs = torch.mean(all_vs)     
            actor_values.append(V)

        return actor_values, total_log_probs, all_obses, all_actions, all_rewards, all_next_obses, all_dones

    def _solve(self, network, dataset_states, verbose=True):
        for epoch in range(self.epochs):
            if epoch % 1000 == 0:
                self.fill_replay(network, dataset_states)

            package = {
                'continue': True,
                'collect_data': False,
                'policy': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'logalpha': self.log_alpha.data
            }

            for p in range(self.nprocs - 1):
                self.comm.send(package, dest=p+1)

            critic_losses = 100.0
            critic_itrs = 0

            while critic_itrs < self.critic_min_iterations_per_epoch or (critic_losses >= self.critic_loss_threshold and critic_itrs < self.critic_max_iterations_per_epoch):
                critic_losses = self.update_critic()
                critic_itrs += 1

            utils.soft_update_params(self.critic, self.critic_target, self.tau)
            utils.soft_update_params(self.actor, self.actor_target, self.tau)

            actor_values, total_log_probs, _, _, _, _, _ = self._solve_once(network, dataset_states)
            actor_values = torch.stack(actor_values, dim=0)
            actor_loss = -torch.mean(actor_values)

            total_log_probs = torch.stack(total_log_probs, dim=0)
            total_log_probs = torch.sum(total_log_probs)

            loss = actor_loss

            self.manual_set_zero_grads(network)

            loss.backward()

            all_actor_loss = [actor_loss]
            
            for p in range(self.nprocs - 1):
                grads = self.comm.recv()

                for name, w in self.actor.named_parameters():
                    w.grad = w.grad + grads['policy_grads'][name]
                
                all_actor_loss.append(grads['actor_loss'])
                total_log_probs += grads['total_log_probs']

                self.add_transitions_to_replay(grads['transitions'])

            all_actor_loss = torch.stack(all_actor_loss, dim=0)

            for name, w in self.actor.named_parameters():
                w.grad = w.grad / self.nprocs
    
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)

            self.actor_optimizer.step()    

            self.update_alpha(total_log_probs)    

            if verbose:
                print(f"Iteration {epoch} ; Actor value = {-torch.mean(all_actor_loss)} ; Critic loss = {critic_losses} ; Alpha = {self.alpha.detach()}")
                sys.stdout.flush()

        package = {
            'continue': False
        }
        for p in range(self.nprocs - 1):
            self.comm.send(package, dest=p+1)
    
    def manual_set_zero_grads(self, network):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        network.optim.zero_grad()

    def fill_replay(self, network, dataset_states):
        warmup_epochs = 2000

        package = {
            'continue': True,
            'collect_data': True,
            'policy': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

        for p in range(self.nprocs - 1):
            self.comm.send(package, dest=p+1)

        with torch.no_grad():
            for _ in range(warmup_epochs):
                _, _, obses, actions, rewards, next_obses, dones = self._solve_once(network, dataset_states)
        
        for p in range(self.nprocs - 1):
            transitions = self.comm.recv()
            self.add_transitions_to_replay(transitions)
    
    def process_fill_replay(self, policy, critic, network, dataset_states):
        warmup_epochs = 2000

        self.actor.load_state_dict(policy)
        self.critic.load_state_dict(critic)

        all_obses = []
        all_actions = []
        all_rewards = []
        all_next_obses = []
        all_dones = []
        
        with torch.no_grad():
            for _ in range(warmup_epochs):
                _, _, obses, actions, rewards, next_obses, dones = self._solve_once(network, dataset_states)
                all_obses.extend(obses)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_next_obses.extend(next_obses)
                all_dones.extend(dones)
        
        transitions = {
            'obses': all_obses,
            'actions': all_actions,
            'rewards': all_rewards,
            'next_obses': all_next_obses,
            'dones': all_dones
        }
            
        return transitions

    def collect_rollouts(self, policy, critic, network, dataset_states):
        self.actor.load_state_dict(policy)
        self.critic.load_state_dict(critic)

        actor_values, total_log_probs, all_obses, all_actions, all_rewards, all_next_obses, all_dones = self._solve_once(network, dataset_states)

        actor_values = torch.stack(actor_values, dim=0)
        actor_loss = -torch.mean(actor_values)

        loss = actor_loss

        self.manual_set_zero_grads(network)

        loss.backward()

        policy_grads = dict()
        
        for name, w in self.actor.named_parameters():
            policy_grads[name] = w.grad
    
        total_log_probs = torch.stack(total_log_probs, dim=0)
        total_log_probs = torch.sum(total_log_probs)

        transitions = {
            'obses': all_obses,
            'actions': all_actions,
            'rewards': all_rewards,
            'next_obses': all_next_obses,
            'dones': all_dones
        }

        return policy_grads, actor_loss, total_log_probs, transitions

    def add_transitions_to_replay(self, transitions):
        obses = transitions['obses']
        actions = transitions['actions']
        rewards = transitions['rewards']
        next_obses = transitions['next_obses']
        dones = transitions['dones']
        for t in range(len(obses)):
            self.replay_buffer.add(obses[t].detach(), actions[t].detach(), rewards[t].detach(), next_obses[t].detach(), dones[t] > 0.5, False)

    def reset(self):
        self.replay_buffer = ArtificialReplayBuffer(obs_shape=[self.obs_dim], action_shape=[self.action_dim], capacity=int(self.capacity), device=self.device)
        #utils.hard_update_params(self.critic, self.critic_target)

    def process_reset(self):
        pass
        #self.actor = DiagGaussianActor(self.obs_dim, self.action_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        #self.critic = DoubleQCritic(self.obs_dim, self.action_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)

    def save(self, PATH):
        sd = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha.data,
            'actor_optim': self.actor_optimizer.state_dict(),
            'critic_optim': self.critic_optimizer.state_dict(),
            'alpha_optim': self.log_alpha_optimizer.state_dict()
        }
        torch.save(sd, PATH)
    
    def load(self, PATH):
        all_dict = torch.load(PATH)
        self.actor.load_state_dict(all_dict['actor'])
        self.critic.load_state_dict(all_dict['critic'])
        self.critic_target.load_state_dict(all_dict['critic'])
        self.log_alpha.data = all_dict['log_alpha']
        self.actor_optimizer.load_state_dict(all_dict['actor_optim'])
        self.critic_optimizer.load_state_dict(all_dict['critic_optim'])
        self.log_alpha_optimizer.load_state_dict(all_dict['alpha_optim'])
