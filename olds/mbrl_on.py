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
from mbopg.actor import SacDiagGaussianActor, DiagGaussianActor
from mbopg.critic import DoubleVCritic
from mbopg.replay import ReplayBuffer
from mbopg.memory.per import LazyPrioritizedMultiStepMemory
from mbopg.artificial_replay_on import ArtificialReplayBuffer


class Policy():

    def __init__(self, actor, action_range, noise_dim, device, std):
        self.actor = actor
        self.action_range = action_range
        self.noise_dim = noise_dim
        self.device = device
        self.std = std
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        
        if sample:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        else:
            action = dist.mean()
            log_prob = None
        
        action = action[0, 0:-self.noise_dim]
        out_action = action * self.action_range[0:-self.noise_dim]
        #action = torch.clamp(action, -self.action_range[0:-self.noise_dim], self.action_range[0:-self.noise_dim])
        return action.detach().cpu().numpy(), out_action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()
    
    def save(self, filepath):
        save_dict = {
            'actor': self.actor,
            'action_range': self.action_range,
            'noise_dim': self.noise_dim,
            'std': self.std
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
        self.std = save_dict['std']


class MBRL_solver(nn.Module):

    def __init__(self, obs_dim, action_dim, horizon, epochs, actor_iterations_per_epoch, actor_repeat_per_epoch,
                 surrogate_epochs,
                 surrogate_batchsize,
                 surrogate_target_update_frequency,
                 action_range, z_range, noise_dim, device,
                 actor_logstd_bounds=[-5, 5], actor_hidden_dim=64, actor_hidden_layers=3, critic_hidden_dim=64, 
                 critic_hidden_layers=3, tau=0.005, surrogate_tau=0.005,
                 actor_lr=1e-4, critic_lr=1e-4, actor_betas=[0.9, 0.999], 
                 critic_betas=[0.9, 0.999], capacity=1e5,
                 gamma=0.99, alpha=0.1, action_noise=0.4, target_noise=0.1, noise_clip=0.2, multi_steps=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim + noise_dim
        self.horizon = horizon
        self.epochs = epochs

        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.multi_steps = multi_steps

        self.actor_logstd_bounds = actor_logstd_bounds
        self.actor_iterations_per_epoch = actor_iterations_per_epoch
        self.actor_repeat_per_epoch = actor_repeat_per_epoch
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

        self.surrogate_epochs = surrogate_epochs
        self.surrogate_batchsize = surrogate_batchsize
        self.surrogate_tau = surrogate_tau
        self.surrogate_target_update_frequency = surrogate_target_update_frequency

        self.gamma = gamma
        self.alpha = alpha

        action_range = list(action_range) + list(z_range)

        if device == 'cuda':
            self.action_range = torch.cuda.FloatTensor(action_range)
        else:
            self.action_range = torch.FloatTensor(action_range)
            
        self.action_range.requires_grad = False

        self.device = device

        self.action_std = torch.tensor([action_noise], device=device)

        self.full_reset()
        
    def solve(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_logprob, dataset_dones, reset=True, verbose=True):
        if reset:
            self.full_reset()
        
        self.reward_dataset_states = []
        self.reward_dataset_logprob = []
        self.reward_dataset_dones = []
        self.reward_dataset_next_states = []
        self.reward_dataset_rewards = []
        self.reward_dataset_actions = []

        self.replay_buffer = ArtificialReplayBuffer(obs_shape=[self.obs_dim], action_shape=[self.action_dim - self.noise_dim], capacity=int(self.capacity), device=self.device)

        with torch.no_grad():
            states = torch.stack(dataset_states, dim=0)
            actions = torch.stack(dataset_actions, dim=0)
            rewards = torch.stack(dataset_rewards, dim=0)
            next_states = torch.stack(dataset_next_states, dim=0)
            logprobs = torch.stack(dataset_logprob, dim=0)
            dones = torch.stack(dataset_dones, dim=0)
            self.replay_buffer.initial_add(states, actions, rewards, next_states, dones, logprobs)

        self._solve(network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_logprob, dataset_dones, verbose=verbose)
        policy = self.actor.state_dict()
        critic = self.value.state_dict()
        return policy, critic
    
    def make_policy(self, policy):
        actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        actor.load_state_dict(policy)
        policy = Policy(actor, self.action_range, self.noise_dim, self.device, self.action_std)
        return policy
    
    def _solve_once(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_logprob, dataset_dones):
        total_log_probs = []

        initial_obs = []
        initial_obs2 = []
        initial_actions = []
        initial_rewards = []
        initial_logprobs = []
        initial_dones = []
        obs = []

        for _ in range(self.actor_iterations_per_epoch):
            random_idx = np.random.randint(0, len(dataset_states))

            obs1 = dataset_states[random_idx]
            obs2 = dataset_next_states[random_idx]

            action = dataset_actions[random_idx]

            log_prob = dataset_logprob[random_idx]
            reward = dataset_rewards[random_idx]
            done = dataset_dones[random_idx]

            initial_logprobs.append(log_prob.clone())
            initial_obs.append(obs1.clone())
            initial_obs2.append(obs2.clone())
            initial_actions.append(action.clone())
            initial_rewards.append(reward.clone())
            initial_dones.append(done.clone())

            for _ in range(self.actor_repeat_per_epoch):
                obs.append(obs1.clone())
        
        initial_obs = torch.stack(initial_obs, dim=0)
        initial_obs2 = torch.stack(initial_obs2, dim=0)
        initial_actions = torch.stack(initial_actions, dim=0)
        initial_rewards = torch.stack(initial_rewards, dim=0)
        initial_logprobs = torch.stack(initial_logprobs, dim=0)
        initial_dones = torch.stack(initial_dones, dim=0)
        obs = torch.stack(obs, dim=0)
   
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        Ws = []
    
        for t in range(self.horizon):
            dist = self.actor(obs)
            action = dist.rsample()

            log_prob = dist.log_prob(action)
            out_action = action * self.action_range.unsqueeze(dim=0)

            act_action = out_action[:, 0:-self.noise_dim]

            z = out_action[:, -self.noise_dim:]

            w = network.sample(1, z)

            next_obs, reward, done = network.primarynet.batch_mbrl(obs, act_action, w)
            
            next_obs = next_obs[:, 0]
            reward = reward[:, 0, 0]
            done = done[:, 0, 0]
            
            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob.sum(-1))
            dones.append(done)
            rewards.append(reward)
            total_log_probs.append(log_prob.sum(-1))
            Ws.append(w)

            self.replay_buffer.add(obs.detach(), action[:, 0:-self.noise_dim].detach(), reward.unsqueeze(dim=-1).detach(), next_obs.detach(),
                                    (done > 0.5).float().unsqueeze(dim=-1).detach(), log_prob[:, 0:-self.noise_dim].detach())

            if t + 1 == self.horizon:
                next_state = next_obs
                dist = self.actor(next_state)
                next_action = dist.rsample()
                next_log_prob = dist.log_prob(next_action).sum(-1)
                break
            
            obs = next_obs

        states.append(next_state)
        actions.append(next_action)
        log_probs.append(next_log_prob)
        
        # Policy
        current_V1, current_V2 = self.value(next_state)
        V = current_V1[:, 0] - self.alpha * next_log_prob
        #for t in reversed(range(len(rewards))):
        #    V = rewards[t] - self.alpha * log_probs[t] + (1.0 - dones[t]) * self.gamma * V
        V = rewards[0] - self.alpha * log_probs[0] + (1.0 - dones[0]) * self.gamma * V
        
        averaged_values = []

        for aidx in range(self.actor_iterations_per_epoch):
            avg_V = 0
            for bidx in range(self.actor_repeat_per_epoch):
                avg_V += V[aidx * self.actor_repeat_per_epoch + bidx]
            avg_V = avg_V / self.actor_repeat_per_epoch
            averaged_values.append(avg_V)
        
        averaged_values = torch.stack(averaged_values, dim=0)
        """
        dist = self.actor(initial_obs)
        initial_actions = torch.cat([initial_actions, torch.zeros((initial_actions.shape[0], self.noise_dim), dtype=torch.float32, device=self.device)], dim=-1)        
        current_logprob = dist.log_prob(initial_actions)[:, 0:-self.noise_dim].sum(-1)
        w = current_logprob - initial_logprobs.sum(-1).detach()
        w = torch.exp(w)
        w = torch.nan_to_num(w)

        dist2 = self.actor(initial_obs2)
        next_action2 = dist2.rsample()
        next_log_prob2 = dist2.log_prob(next_action2).sum(-1)
        current2_V1, current2_V2 = self.value(initial_obs2)
        V2 = current2_V1[:, 0] - self.alpha * next_log_prob2

        initial_rewards = initial_rewards.squeeze(dim=-1)
        initial_dones = initial_dones.squeeze(dim=-1)

        sim_values = (initial_rewards.detach() + self.gamma * (1.0 - initial_dones.detach()) * V2)

        mean_values = 0.5 * (sim_values + averaged_values).detach()
        averaged_values = averaged_values - mean_values
        sim_values = sim_values - mean_values
        sim_values = w * sim_values"""
    
        actor_values = averaged_values#torch.cat([averaged_values, sim_values], dim=0)
        
        return actor_values

    def _solve(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_logprob, dataset_dones, verbose=True):
        for epoch in range(self.epochs):
            value_loss = self.train_value()
            
            if (epoch + 1) % self.surrogate_target_update_frequency == 0:
                utils.soft_update_params(self.value, self.value_target, self.surrogate_tau)
                actor_values = self._solve_once(network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_logprob, dataset_dones)

                loss = -torch.mean(actor_values)
                self.manual_set_zero_grads(network)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
                self.actor_optimizer.step()

                if verbose:
                    print(f"Iteration {epoch} ; Actor value = {-loss} ;  Value loss = {value_loss}")
                    sys.stdout.flush()

    def train_value(self):
        obs, action, reward, next_obs, done, logprob = self.replay_buffer.sample(self.surrogate_batchsize)

        target_V1, target_V2 = self.value_target(next_obs)
        target_V = torch.min(target_V1, target_V2)
        target_V = reward + (1.0 - done) * self.gamma * target_V
        target_V = target_V.detach()

        dist = self.actor(obs)
        action = torch.cat([action, torch.ones((action.shape[0], self.noise_dim), dtype=torch.float32, device=self.device)], dim=-1)
        current_logprob = dist.log_prob(action)[:, 0:-self.noise_dim].sum(-1)
        w = current_logprob.detach() - logprob.sum(-1)
        w = torch.exp(w).unsqueeze(dim=-1)
        wsum = torch.sum(w, dim=0)
        w = w / wsum
        w = torch.nan_to_num(w)
        #w = w.unsqueeze(dim=-1)
        
        current_V1, current_V2 = self.value(obs)
        #value_loss1 = w * (current_V1 - target_V).pow(2)
        #value_loss1 = w + torch.log((current_V1 - target_V).pow(2))
        #value_loss2 = w + torch.log((current_V2 - target_V).pow(2))
        value_loss1 = w * (current_V1 - target_V).pow(2)
        value_loss2 = w * (current_V2 - target_V).pow(2)
        value_loss = 0.5 * (torch.sum(value_loss1) + torch.sum(value_loss2))

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 5.0)
        self.value_optimizer.step()
        
        return value_loss

    def manual_set_zero_grads(self, network):
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        network.optim.zero_grad()

    def solve_surrogate(self, network, augment=True):
        #aug_replay_buffer = LazyPrioritizedMultiStepMemory(self.capacity, self.obs_dim, self.action_dim - self.noise_dim, self.device, beta_steps=self.surrogate_epochs, multi_step=self.multi_steps)
        aug_replay_buffer = ReplayBuffer(obs_shape=[self.obs_dim], action_shape=[self.action_dim - self.noise_dim], capacity=int(self.capacity), device=self.device)

        for idx in range(len(self.surrogate_replay_buffer)):
            state = self.surrogate_replay_buffer.obses[idx]
            action = self.surrogate_replay_buffer.actions[idx]
            reward = self.surrogate_replay_buffer.rewards[idx]
            next_state = self.surrogate_replay_buffer.next_obses[idx]
            done = 1.0 - self.surrogate_replay_buffer.not_dones[idx]
            done_no_max = 1.0 - self.surrogate_replay_buffer.not_dones_no_max[idx]
            if idx + 1 == len(self.surrogate_replay_buffer):
                done_no_max = True
            #aug_replay_buffer.append(state, action, reward, next_state, done, done_no_max)
            aug_replay_buffer.add(state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    done_no_max)

        if augment:
            with torch.no_grad():
                for idx in range(0, len(self.surrogate_replay_buffer), self.surrogate_batchsize):
                    initial_obs = []
                    for j in range(self.surrogate_batchsize):
                        init_obs = torch.from_numpy(self.surrogate_replay_buffer.obses[idx + j]).to(self.device)
                        for _ in range(self.actor_repeat_per_epoch):
                            initial_obs.append(init_obs)
                    obs = torch.stack(initial_obs, dim=0)

                    obs_buffer = []
                    action_buffer = []
                    reward_buffer = []
                    next_obs_buffer = []
                    done_buffer = []
                    done_no_max_buffer = []

                    for t in range(self.horizon):
                        dist = self.actor(obs)
                        action = dist.sample()

                        log_prob = dist.log_prob(action).sum(-1)
                        out_action = action * self.action_range.unsqueeze(dim=0)

                        act_action = out_action[:, 0:-self.noise_dim]

                        z = out_action[:, -self.noise_dim:]

                        w = network.sample(1, z)

                        next_obs, reward, done = network.primarynet.batch_mbrl(obs, act_action, w)
                        
                        next_obs = next_obs[:, 0]
                        reward = reward[:, 0, 0]
                        done = done[:, 0, 0] > 0.5
                        done_no_max = done

                        if t + 1 == self.horizon:
                            done_no_max = torch.ones_like(done_no_max)
                        
                        obs_buffer.append(obs.cpu().numpy())
                        action_buffer.append(action[:, 0:-self.noise_dim].cpu().numpy())
                        reward_buffer.append(reward.cpu().numpy())
                        next_obs_buffer.append(next_obs.cpu().numpy())
                        done_buffer.append(done.float().cpu().numpy())
                        done_no_max_buffer.append(done_no_max.float().cpu().numpy())

                        obs = next_obs

                    for jdx in range(self.surrogate_batchsize * self.actor_repeat_per_epoch):
                        for t in range(self.horizon):
                            """
                            aug_replay_buffer.append(obs_buffer[t][jdx],
                                                    action_buffer[t][jdx], 
                                                    reward_buffer[t][jdx], 
                                                    next_obs_buffer[t][jdx], 
                                                    done_buffer[t][jdx], 
                                                    done_no_max_buffer[t][jdx])
                            """
                            aug_replay_buffer.add(obs_buffer[t][jdx],
                                                    action_buffer[t][jdx], 
                                                    reward_buffer[t][jdx], 
                                                    next_obs_buffer[t][jdx], 
                                                    done_buffer[t][jdx], 
                                                    done_no_max_buffer[t][jdx])
                            if done_buffer[t][jdx]:
                                break
        
        for epoch in range(self.surrogate_epochs):
            #atch, weights = aug_replay_buffer.sample(self.surrogate_batchsize)
            #obs, action, reward, next_obs, done = batch
            obs, action, reward, next_obs, not_done, _ = aug_replay_buffer.sample(self.surrogate_batchsize)

            with torch.no_grad():
                pi_targ = self.surrogate_actor_target(next_obs)
                epsilon = torch.randn_like(pi_targ) * self.target_noise
                epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
                next_action = pi_targ + epsilon
                next_action = torch.clamp(next_action, -self.action_range[0:-self.noise_dim], self.action_range[0:-self.noise_dim])
            
            target_Q1, target_Q2 = self.surrogate_critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_V
            target_Q = target_Q.detach()

            current_Q1, current_Q2 = self.surrogate_critic(obs, action)
            errors = torch.abs(current_Q1.detach() - target_Q)
            critic_loss = torch.mean((current_Q1 - target_Q).pow(2)) + torch.mean((current_Q2 - target_Q).pow(2))
    
            self.surrogate_critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.surrogate_critic.parameters(), 50.0)
            self.surrogate_critic_optimizer.step()

            if (epoch + 1) % self.surrogate_target_update_frequency == 0:
                utils.soft_update_params(self.surrogate_critic, self.surrogate_critic_target, self.surrogate_tau)
                utils.soft_update_params(self.surrogate_actor, self.surrogate_actor_target, self.surrogate_tau)

                obs, action, reward, next_obs, not_done, _ = aug_replay_buffer.sample(self.surrogate_batchsize)

                action = self.surrogate_actor(obs)
                actor_Q1, actor_Q2 = self.surrogate_critic(obs, action)
                actor_loss = -(actor_Q1).mean()

                self.surrogate_actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.surrogate_actor.parameters(), 5.0)
                self.surrogate_actor_optimizer.step()
                
                print(f"Surrogate Iteration {epoch} ; Actor loss = {actor_loss} ; Critic loss = {critic_loss}")
                sys.stdout.flush()

    def full_reset(self):
        self.value = DoubleVCritic(self.obs_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.value_target = DoubleVCritic(self.obs_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_target.requires_grad = False
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.critic_lr, betas=self.critic_betas)

        self.actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)

    def save(self, PATH):
        sd = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optimizer.state_dict()
        }
        torch.save(sd, PATH)
    
    def load(self, PATH):
        all_dict = torch.load(PATH)
        self.actor.load_state_dict(all_dict['actor'])
        self.actor_optimizer.load_state_dict(all_dict['actor_optim'])
