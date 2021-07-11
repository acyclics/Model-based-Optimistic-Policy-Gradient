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
from mbopg.actor import DeterministicActor
from mbopg.critic import DoubleQCritic
from mbopg.replay import ReplayBuffer


class Policy():

    def __init__(self, actor, action_range, noise_dim, act_std, device):
        self.actor = actor
        self.action_range = action_range
        self.noise_dim = noise_dim
        self.act_std = act_std
        self.device = device
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        action, noise_mu = self.actor(obs)
        action = action.detach()

        if sample:
            action += torch.randn_like(action) * self.act_std
        
        action = action[0]
        action = action * self.action_range[0:-self.noise_dim]
        
        return action.detach().cpu().numpy()
    
    def eval_act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        action, noise_mu = self.actor(obs)
        action = action.detach()

        if sample:
            action += torch.randn_like(action) * self.act_std
        
        action = action[0]
        action = action * self.action_range[0:-self.noise_dim]
        
        return action.detach().cpu().numpy()

    def save(self, filepath):
        save_dict = {
            'actor': self.actor,
            'action_range': self.action_range,
            'noise_dim': self.noise_dim,
            'act_std': self.act_std
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
        self.act_std = save_dict['act_std']


class MBRL_solver(nn.Module):

    def __init__(self, obs_dim, action_dim, horizon, epochs, actor_iterations_per_epoch,
                 target_update_frequency,
                 action_range, z_range, noise_dim, device,
                 actor_logstd_bounds=[-2, 2], actor_hidden_dim=64, actor_hidden_layers=3, critic_hidden_dim=64, 
                 critic_hidden_layers=3, tau=0.005,
                 actor_lr=1e-4, critic_lr=1e-4, actor_betas=[0.9, 0.999], 
                 critic_betas=[0.9, 0.999], capacity=1e5,
                 gamma=0.99, alpha=0.1, noise_alpha=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim + noise_dim
        self.horizon = horizon
        self.epochs = epochs

        self.actor_logstd_bounds = actor_logstd_bounds
        self.actor_iterations_per_epoch = actor_iterations_per_epoch
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
        self.target_update_frequency = target_update_frequency
        
        self.log_alpha = torch.tensor(np.log(alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.noise_alpha = noise_alpha

        self.target_entropy = -action_dim

        action_range = list(action_range) + list(z_range)

        if device == 'cuda':
            self.action_range = torch.cuda.FloatTensor(action_range)
        else:
            self.action_range = torch.FloatTensor(action_range)
            
        self.action_range.requires_grad = False

        self.device = device
        self.full_reset()
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def solve(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob, reset=True, verbose=True):
        if reset:
            self.full_reset()

        self._solve(network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob, verbose=verbose)

        policy = self.actor.state_dict()
        critic = self.critic.state_dict()

        return policy, critic
    
    def make_policy(self, policy):
        actor = DeterministicActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers).to(self.device)
        actor.load_state_dict(policy)
        policy = Policy(actor, self.action_range, self.noise_dim, 0.1, self.device)
        return policy
    
    def _get_initial_data(self, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob):
        with torch.no_grad():
            initial_obs1 = []
            initial_obs2 = []
            initial_act = []
            initial_rewards = []
            initial_dones = []
            initial_logprob = []

            for _ in range(self.actor_iterations_per_epoch):
                ridx = np.random.randint(0, len(dataset_next_states))

                init_obs1 = dataset_states[ridx]
                init_obs2 = dataset_next_states[ridx]
                act = dataset_actions[ridx]
                rew = dataset_rewards[ridx]
                d = dataset_dones[ridx]

                initial_obs1.append(init_obs1)
                initial_obs2.append(init_obs2)
                initial_act.append(act)
                initial_rewards.append(rew)
                initial_dones.append(d)

            obs = torch.stack(initial_obs2, dim=0)
            initial_obs = torch.stack(initial_obs1, dim=0)
            initial_obs2 = torch.stack(initial_obs2, dim=0)
            initial_act = torch.stack(initial_act, dim=0)
            initial_rewards = torch.stack(initial_rewards, dim=0)
            initial_dones = torch.stack(initial_dones, dim=0)

            return obs, initial_obs, initial_obs2, initial_act, initial_rewards, initial_dones, initial_logprob

    def _rollout(self, network, obs):
        states = []
        actions = []
        act_log_probs = []
        noise_log_probs = []
        total_log_probs = []
        rewards = []
        dones = []
        Ws = []

        for t in range(self.horizon):
            act_action, noise_action = self.actor(obs)

            action += torch.randn_like(action) * self.act_std

            action = act_action * self.action_range[0:-self.noise_dim].unsqueeze(dim=0)
            z = noise_action * self.action_range[-self.noise_dim:].unsqueeze(dim=0)

            w = network.sample(1, z)

            next_obs, reward, done = network.primarynet.batch_mbrl(obs, action, w)
            
            next_obs = next_obs[:, 0]
            reward = reward[:, 0, 0]
            done = done[:, 0, 0]
            
            states.append(obs)
            actions.append(act_action)
            dones.append(done)
            rewards.append(reward)
            Ws.append(w)

            if t + 1 == self.horizon:
                break
            
            obs = next_obs

        return states, actions, rewards, dones, next_obs

    def _train_critic(self, data, rollout):
        obs, initial_obs, initial_obs2, initial_act, initial_rewards, initial_dones, initial_logprob = data
        states, actions, rewards, dones, next_obs = rollout

        target_noise = 0.1
        noise_clip = 0.2
        act_limit = 1.0

        with torch.no_grad():
            pi_targ, _ = self.actor_target(next_obs)
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            next_action = pi_targ + epsilon
            next_action = torch.clamp(next_action, -act_limit, act_limit)

        critic_loss1, critic_loss2 = [], []
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        return_expansion = torch.min(target_Q1, target_Q2)
        
        for t in reversed(range(len(rewards))):
            return_expansion = rewards[t].unsqueeze(dim=-1) + (1.0 - dones[t].unsqueeze(dim=-1)) * self.gamma * return_expansion
    
        target_Q = initial_rewards + (1 - initial_dones) * self.gamma * return_expansion
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(initial_obs, initial_act)
        err1 = (current_Q1 - target_Q).pow(2)
        err2 = (current_Q2 - target_Q).pow(2)
        critic_loss1.append(err1)
        critic_loss2.append(err2)

        critic_loss1 = torch.stack(critic_loss1, dim=1)
        critic_loss1 = torch.mean(critic_loss1, dim=1)
        critic_loss2 = torch.stack(critic_loss2, dim=1)
        critic_loss2 = torch.mean(critic_loss2, dim=1)
        critic_loss = torch.mean(critic_loss1) + torch.mean(critic_loss2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        return critic_loss

    def _train_actor(self, network, data, rollout):
        obs, initial_obs, initial_obs2, initial_act, initial_rewards, initial_dones, initial_logprob = data
        states, actions, rewards, dones, next_obs = rollout

        next_action, _ = self.actor(next_obs)
        noise_actor_Q1, noise_actor_Q2 = self.critic(next_obs, next_action)
        noise_actor_V = torch.min(noise_actor_Q1, noise_actor_Q2)
        noise_return_expansion = noise_actor_V
        
        for t in reversed(range(len(rewards))):
            noise_return_expansion = rewards[t].unsqueeze(dim=-1) + (1.0 - dones[t].unsqueeze(dim=-1)) * self.gamma * noise_return_expansion

        noise_return_expansion = initial_rewards + self.gamma * (1 - initial_dones) * noise_return_expansion
        noise_actor_loss = -noise_return_expansion.mean()

        all_states = torch.cat(states, dim=0)
        all_states = torch.cat([initial_obs, all_states, next_obs], dim=0)
        all_states = all_states.detach()
        action, _ = self.actor(all_states)
        actor_Q, _ = self.critic(all_states, action)
        actor_loss = (-actor_Q).mean()

        total_actor_loss = noise_actor_loss + actor_loss

        self.actor_optimizer_noise.zero_grad()
        self.actor_optimizer_act.zero_grad()
        network.optim.zero_grad()

        total_actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.trunk_noise.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.actor.trunk_act.parameters(), 5.0)
        self.actor_optimizer_noise.step()
        self.actor_optimizer_act.step()

        return actor_loss

    def _solve(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob, verbose=True):
        for epoch in range(self.epochs):
            data = self._get_initial_data(dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob)            
            rollout = self._rollout(network, data[0])

            critic_loss = self._train_critic(data, rollout)
            
            if (epoch + 1) % self.target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target, self.tau)
                utils.soft_update_params(self.actor, self.actor_target, self.tau)

                actor_loss = self._train_actor(network, data, rollout)

                if verbose:
                    print(f"Iteration {epoch} ; Actor value = {-actor_loss} ; Critic loss = {critic_loss}")
                    sys.stdout.flush()

    def full_reset(self):
        self.actor = DeterministicActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers).to(self.device)
        self.actor_target = DeterministicActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad = False

        self.actor_optimizer_act = torch.optim.Adam(self.actor.trunk_act.parameters(), lr=self.actor_lr, betas=self.actor_betas)
        self.actor_optimizer_noise = torch.optim.Adam(self.actor.trunk_noise.parameters(), lr=self.actor_lr, betas=self.actor_betas)

        self.critic = DoubleQCritic(self.obs_dim, self.action_dim - self.noise_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target = DoubleQCritic(self.obs_dim, self.action_dim - self.noise_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=1e-4,
                                                    betas=[0.9, 0.999])

    def reset_noise(self):
        new_actor = DeterministicActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers).to(self.device)
        self.actor.trunk_noise.load_state_dict(new_actor.trunk_noise.state_dict())
        self.actor_optimizer_noise = torch.optim.Adam(self.actor.trunk_noise.parameters(), lr=self.actor_lr, betas=self.actor_betas)

    def save(self, PATH):
        sd = {
            'actor': self.actor.state_dict(),
            'actor_optim_act': self.actor_optimizer_act.state_dict(),
            'actor_optim_noise': self.actor_optimizer_noise.state_dict()
        }
        torch.save(sd, PATH)
    
    def load(self, PATH):
        all_dict = torch.load(PATH)
        self.actor.load_state_dict(all_dict['actor'])
        self.actor_optimizer_act.load_state_dict(all_dict['actor_optim_act'])
        self.actor_optimizer_noise.load_state_dict(all_dict['actor_optim_noise'])
