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
from mbopg.actor_algo6 import SacDiagGaussianActor
from mbopg.critic import DoubleQCritic
from mbopg.replay import ReplayBuffer


class Policy():

    def __init__(self, actor, action_range, noise_dim, device):
        self.actor = actor
        self.action_range = action_range
        self.noise_dim = noise_dim
        self.device = device
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        act_dist, noise_dist = self.actor(obs)

        if sample:
            action = act_dist.sample()
            log_prob = act_dist.log_prob(action)[0]
            log_prob = log_prob.detach().cpu().numpy()
        else:
            action = act_dist.mean
            log_prob = None
        
        action = action[0]
        action = action * self.action_range[0:-self.noise_dim]
        
        return action.detach().cpu().numpy(), log_prob
    
    def eval_act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        act_dist, noise_dist = self.actor(obs)

        if sample:
            action = act_dist.sample()
        else:
            action = act_dist.mean
        
        action = action[0]
        action = action * self.action_range[0:-self.noise_dim]
        
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
        actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        actor.load_state_dict(policy)
        policy = Policy(actor, self.action_range, self.noise_dim, self.device)
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
                lp = dataset_logprob[ridx]

                initial_obs1.append(init_obs1)
                initial_obs2.append(init_obs2)
                initial_act.append(act)
                initial_rewards.append(rew)
                initial_dones.append(d)
                initial_logprob.append(lp)

            initial_obs = torch.stack(initial_obs1, dim=0)
            initial_obs2 = torch.stack(initial_obs2, dim=0)
            initial_act = torch.stack(initial_act, dim=0)
            initial_rewards = torch.stack(initial_rewards, dim=0)
            initial_dones = torch.stack(initial_dones, dim=0)
            initial_logprob = torch.stack(initial_logprob, dim=0)

            return initial_obs, initial_obs2, initial_act, initial_rewards, initial_dones, initial_logprob

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
            act_dist, noise_dist = self.actor(obs)

            act_action = act_dist.rsample()
            noise_action = noise_dist.rsample()

            action = act_action * self.action_range[0:-self.noise_dim].unsqueeze(dim=0)
            z = noise_action * self.action_range[-self.noise_dim:].unsqueeze(dim=0)

            act_log_prob = act_dist.log_prob(act_action).sum(-1)
            noise_log_prob = noise_dist.log_prob(noise_action).sum(-1)

            w = network.sample(1, z)

            next_obs, reward, done = network.primarynet.batch_mbrl(obs, action, w)
            
            next_obs = next_obs[:, 0]
            reward = reward[:, 0, 0]
            done = done[:, 0, 0]
            
            states.append(obs)
            actions.append(act_action)
            act_log_probs.append(act_log_prob)
            noise_log_probs.append(noise_log_prob)
            dones.append(done)
            rewards.append(reward)
            total_log_probs.append(act_log_prob)
            Ws.append(w)

            if t + 1 == self.horizon:
                act_dist, noise_dist = self.actor(next_obs)
                next_action = act_dist.rsample()
                next_act_log_prob = act_dist.log_prob(next_action).sum(-1)
                next_noise = noise_dist.rsample()
                next_noise_log_prob = noise_dist.log_prob(next_noise).sum(-1)
                break
            
            obs = next_obs

        return states, actions, act_log_probs, noise_log_probs, total_log_probs, rewards, dones, next_obs, next_action, next_act_log_prob, next_noise, next_noise_log_prob

    def _train_critic(self, data, rollout1, rollout2):
        initial_obs, initial_obs2, initial_act, initial_rewards, initial_dones, initial_logprob = data
        states1, actions1, act_log_probs1, _, _, rewards1, dones1, next_obs1, next_action1, next_act_log_prob1, _, _ = rollout1
        states2, actions2, act_log_probs2, _, _, rewards2, dones2, next_obs2, next_action2, next_act_log_prob2, _, _ = rollout2

        critic_loss1, critic_loss2 = [], []

        target_Q1, target_Q2 = self.critic_target(next_obs1.detach(), next_action1.detach())
        target_V1 = torch.min(target_Q1, target_Q2)
        return_expansion1 = target_V1.detach() - self.alpha.detach() * next_act_log_prob1.unsqueeze(dim=-1)
        for t in reversed(range(len(rewards1))):
            return_expansion1 = rewards1[t].unsqueeze(dim=-1) - self.alpha.detach() * act_log_probs1[t].unsqueeze(dim=-1) + (1.0 - dones1[t].unsqueeze(dim=-1)) * self.gamma * return_expansion1
        return_expansion1 = return_expansion1.detach()

        actions = torch.stack(actions1, dim=1)[:, 0].detach()
        current_Q1, current_Q2 = self.critic(initial_obs, actions)
        err1 = (current_Q1 - return_expansion1).pow(2)
        err2 = (current_Q2 - return_expansion1).pow(2)
        critic_loss1.append(err1)
        critic_loss2.append(err2)

        target_Q1, target_Q2 = self.critic_target(next_obs2.detach(), next_action2.detach())
        target_V2 = torch.min(target_Q1, target_Q2)
        return_expansion2 = target_V2.detach() - self.alpha.detach() * next_act_log_prob2.unsqueeze(dim=-1)
        for t in reversed(range(len(rewards2))):
            return_expansion2 = rewards2[t].unsqueeze(dim=-1) - self.alpha.detach() * act_log_probs2[t].unsqueeze(dim=-1) + (1.0 - dones2[t].unsqueeze(dim=-1)) * self.gamma * return_expansion2
        return_expansion2 = initial_rewards + self.gamma * (1 - initial_dones) * return_expansion2
        return_expansion2 = return_expansion2.detach()

        current_Q1, current_Q2 = self.critic(initial_obs, initial_act)
        err1 = (current_Q1 - return_expansion2).pow(2)
        err2 = (current_Q2 - return_expansion2).pow(2)
        critic_loss1.append(err1)
        critic_loss2.append(err2)

        critic_loss1 = torch.cat(critic_loss1, dim=0)
        critic_loss1 = torch.mean(critic_loss1)
        critic_loss2 = torch.cat(critic_loss2, dim=0)
        critic_loss2 = torch.mean(critic_loss2)
        critic_loss = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        return critic_loss

    def _train_actor(self, network, data, rollout):
        initial_obs, initial_obs2, initial_act, initial_rewards, initial_dones, initial_logprob = data
        states, actions, act_log_probs, noise_log_probs, total_log_probs, rewards, dones, next_obs, next_action, next_act_log_prob, next_noise, next_noise_log_prob = rollout

        noise_actor_Q1, noise_actor_Q2 = self.critic(next_obs, next_action)
        noise_actor_V = torch.min(noise_actor_Q1, noise_actor_Q2)
        noise_return_expansion = noise_actor_V# - self.noise_alpha * next_noise_log_prob.unsqueeze(dim=-1)
        
        for t in reversed(range(len(rewards))):
            noise_return_expansion = rewards[t].unsqueeze(dim=-1) + (1.0 - dones[t].unsqueeze(dim=-1)) * self.gamma * noise_return_expansion

        noise_actor_loss = -noise_return_expansion.mean()

        #all_states = torch.cat(states, dim=0)
        #all_states = torch.cat([initial_obs, all_states, next_obs], dim=0)
        all_states = initial_obs#all_states.detach()
        act_dist, noise_dist = self.actor(all_states)
        action = act_dist.rsample()
        log_prob = act_dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(all_states, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        total_actor_loss = noise_actor_loss + actor_loss

        self.actor_optimizer_noise.zero_grad()
        self.actor_optimizer_act.zero_grad()
        network.optim.zero_grad()

        total_actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.trunk_noise.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.actor.trunk_act.parameters(), 5.0)
        self.actor_optimizer_noise.step()
        self.actor_optimizer_act.step()

        total_log_probs = torch.cat(total_log_probs, dim=0).unsqueeze(dim=-1)
        total_log_probs = torch.cat([total_log_probs, log_prob], dim=0)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-total_log_probs - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 5.0)
        self.log_alpha_optimizer.step()

        return actor_loss

    def _solve(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob, verbose=True):
        for epoch in range(self.epochs):
            data = self._get_initial_data(dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob)            
            rollout1 = self._rollout(network, data[0])
            rollout2 = self._rollout(network, data[1])

            critic_loss = self._train_critic(data, rollout1, rollout2)
            
            if (epoch + 1) % self.target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target, self.tau)

                actor_loss = self._train_actor(network, data, rollout1)

                if verbose:
                    print(f"Iteration {epoch} ; Actor value = {-actor_loss} ; Critic loss = {critic_loss}")
                    sys.stdout.flush()

    def full_reset(self):
        self.actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        
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
        new_actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
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
