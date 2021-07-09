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
from mbopg.critic import DoubleQCritic
from mbopg.replay import ReplayBuffer
from mbopg.memory.per import LazyPrioritizedMultiStepMemory
from mbopg.artificial_replay import ArtificialReplayBuffer


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
        act_dist, noise_dist = self.actor(obs)

        if sample:
            action = act_dist.sample()
            log_prob = act_dist.log_prob(action)[0]
        else:
            action = act_dist.mean
            log_prob = None
        
        action = action[0]
        #action = action[0:-self.noise_dim] * self.action_range[0:-self.noise_dim]
        action = action * self.action_range[0:-self.noise_dim]
        
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()
    
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

        self.surrogate_replay_buffer = ReplayBuffer(obs_shape=[self.obs_dim], action_shape=[self.action_dim - noise_dim], capacity=int(self.capacity), device=self.device)

        self.action_std = torch.tensor([action_noise], device=device)

        self.full_reset()
        
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
        policy = Policy(actor, self.action_range, self.noise_dim, self.device, self.action_std)
        return policy
    
    def _solve_once_critic(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones):
        total_log_probs = []
        initial_obs1 = []
        initial_obs2 = []
        initial_act = []
        initial_rewards = []
        initial_dones = []

        for _ in range(self.actor_iterations_per_epoch):
            ridx = np.random.randint(0, len(dataset_next_states))

            init_obs1 = dataset_states[ridx]
            init_obs2 = dataset_next_states[ridx]
            act = dataset_actions[ridx]
            rew = dataset_rewards[ridx]
            d = dataset_dones[ridx]

            initial_obs2.append(init_obs2.clone())            
            initial_obs1.append(init_obs1)
            initial_act.append(act)
            initial_rewards.append(rew)
            initial_dones.append(d)

        obs = torch.stack(initial_obs2, dim=0)
        initial_obs = torch.stack(initial_obs1, dim=0)
        initial_act = torch.stack(initial_act, dim=0)
        initial_rewards = torch.stack(initial_rewards, dim=0)
        initial_dones = torch.stack(initial_dones, dim=0)
                        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        Ws = []
    
        for t in range(self.horizon):
            act_dist, noise_dist = self.actor(obs)

            act_action = act_dist.rsample()
            noise_action = noise_dist.rsample()

            action = act_action * self.action_range[0:-self.noise_dim].unsqueeze(dim=0)
            z = noise_action * self.action_range[self.noise_dim:].unsqueeze(dim=0)

            log_prob = act_dist.log_prob(act_action).sum(-1)

            w = network.sample(1, z)

            next_obs, reward, done = network.primarynet.batch_mbrl(obs, action, w)
            
            next_obs = next_obs[:, 0]
            reward = reward[:, 0, 0]
            done = done[:, 0, 0]
            
            states.append(obs)
            actions.append(act_action)
            log_probs.append(log_prob)
            dones.append(done)
            rewards.append(reward)
            total_log_probs.append(log_prob)
            Ws.append(w)

            if t + 1 == self.horizon:
                act_dist, noise_dist = self.actor(next_obs)
                next_action = act_dist.rsample()
                next_log_prob = act_dist.log_prob(next_action).sum(-1)
                break
            
            obs = next_obs

        critic_loss = []

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2)
        return_expansion = target_V.detach() - self.alpha * next_log_prob.unsqueeze(dim=-1)
        
        for t in reversed(range(len(rewards))):
            return_expansion = rewards[t].unsqueeze(dim=-1) - self.alpha * log_probs[t].unsqueeze(dim=-1) + (1.0 - dones[t].unsqueeze(dim=-1)) * self.gamma * return_expansion

            s = states[t].detach()
            a = actions[t].detach()
            ret = return_expansion.detach()

            current_Q1, current_Q2 = self.critic(s, a)
            closs = (current_Q1 - ret).pow(2) + (current_Q2 - ret).pow(2)
            critic_loss.append(closs)
        
        return_expansion = initial_rewards + self.gamma * (1 - initial_dones) * return_expansion
        ret = return_expansion.detach()

        current_Q1, current_Q2 = self.critic(initial_obs, initial_act)
        closs = (current_Q1 - ret).pow(2) + (current_Q2 - ret).pow(2)
        critic_loss.append(closs)

        critic_loss = torch.stack(critic_loss, dim=1)
        critic_loss = torch.mean(critic_loss, dim=1)
        critic_loss = torch.mean(critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        return critic_loss

    def _solve_once_actor(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob):
        total_log_probs = []
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

        obs = torch.stack(initial_obs1, dim=0)
        initial_obs = torch.stack(initial_obs1, dim=0)
        initial_obs2 = torch.stack(initial_obs2, dim=0)
        initial_act = torch.stack(initial_act, dim=0)
        initial_rewards = torch.stack(initial_rewards, dim=0)
        initial_dones = torch.stack(initial_dones, dim=0)
        initial_logprob = torch.stack(initial_logprob, dim=0)
                        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        Ws = []
    
        for t in range(self.horizon):
            act_dist, noise_dist = self.actor(obs)

            act_action = act_dist.rsample()
            noise_action = noise_dist.rsample()

            action = act_action * self.action_range[0:-self.noise_dim].unsqueeze(dim=0)
            z = noise_action * self.action_range[self.noise_dim:].unsqueeze(dim=0)

            log_prob = act_dist.log_prob(act_action).sum(-1)

            w = network.sample(1, z)

            next_obs, reward, done = network.primarynet.batch_mbrl(obs, action, w)
            
            next_obs = next_obs[:, 0]
            reward = reward[:, 0, 0]
            done = done[:, 0, 0]
            
            states.append(obs)
            actions.append(act_action)
            log_probs.append(log_prob)
            dones.append(done)
            rewards.append(reward)
            total_log_probs.append(log_prob)
            Ws.append(w)

            if t + 1 == self.horizon:
                act_dist, noise_dist = self.actor(next_obs)
                next_action = act_dist.rsample()
                next_log_prob = act_dist.log_prob(next_action).sum(-1)
                break
            
            obs = next_obs

        actor_Q1, actor_Q2 = self.critic(next_obs, next_action)
        actor_V = torch.min(actor_Q1, actor_Q2)
        return_expansion = actor_V - self.alpha * next_log_prob.unsqueeze(dim=-1)
        
        for t in reversed(range(len(rewards))):
            return_expansion = rewards[t].unsqueeze(dim=-1) - self.alpha * log_probs[t].unsqueeze(dim=-1) + (1.0 - dones[t].unsqueeze(dim=-1)) * self.gamma * return_expansion

        #return_expansion = initial_rewards + self.gamma * (1 - initial_dones) * return_expansion
        
        """
        act_dist, noise_dist = self.actor(initial_obs)
        current_logprob = act_dist.log_prob(initial_act)
        w = current_logprob.sum(-1) - initial_logprob.sum(-1)
        w = torch.exp(w)
        w = w.unsqueeze(dim=-1)
        w = torch.nan_to_num(w, posinf=0.0, neginf=0.0)
    
        return_expansion = w * return_expansion
        """
        act_dist, noise_dist = self.actor(initial_obs2)
        next_action2 = act_dist.sample()
        next_Q1, next_Q2 = self.critic(initial_obs2, next_action2)
        next_V = torch.min(next_Q1, next_Q2)
        baseline = initial_rewards + self.gamma * (1 - initial_dones) * next_V

        return_expansion = return_expansion - baseline.detach()

        actor_loss = -torch.mean(return_expansion)
        
        self.manual_set_zero_grads(network)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        return actor_loss

    def _solve(self, network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob, verbose=True):
        for epoch in range(self.epochs):
            critic_loss = self._solve_once_critic(network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones)

            if (epoch + 1) % self.surrogate_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target, self.tau)
                actor_loss = self._solve_once_actor(network, dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones, dataset_logprob)
  
                if verbose:
                    print(f"Iteration {epoch} ; Actor value = {-actor_loss} ; Critic loss = {critic_loss}")
                    sys.stdout.flush()

    def manual_set_zero_grads(self, network):
        self.actor_optimizer.zero_grad()
        network.optim.zero_grad()

    def solve_surrogate(self, network, augment=True):
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
        self.actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas)

        self.critic = DoubleQCritic(self.obs_dim, self.action_dim - self.noise_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target = DoubleQCritic(self.obs_dim, self.action_dim - self.noise_dim, self.critic_hidden_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)

    def half_reset(self):
        self.actor = SacDiagGaussianActor(self.obs_dim, self.action_dim, self.noise_dim, self.actor_hidden_dim, self.actor_hidden_layers, self.actor_logstd_bounds).to(self.device)
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


"""
def _solve_once_actor(self, network, dataset_states):
        total_log_probs = []
        initial_obs1 = []
       
        for _ in range(self.actor_iterations_per_epoch):
            ridx = np.random.randint(0, len(dataset_states))
            init_obs1 = dataset_states[ridx]
            for _ in range(self.actor_repeat_per_epoch):
                initial_obs1.append(init_obs1.clone())
            
        obs = torch.stack(initial_obs1, dim=0)
                        
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
            log_probs.append(log_prob)
            dones.append(done)
            rewards.append(reward)
            total_log_probs.append(log_prob)
            Ws.append(w)

            if t + 1 == self.horizon:
                dist = self.actor(next_obs)
                next_action = dist.rsample()
                next_log_prob = dist.log_prob(next_action).sum(-1)
                break
            
            obs = next_obs

        actor_Q1, actor_Q2 = self.critic(next_obs, next_action[:, 0:-self.noise_dim])
        return_expansion = torch.min(actor_Q1, actor_Q2) - self.alpha * next_log_prob.unsqueeze(dim=-1)
        
        for t in reversed(range(len(rewards))):
            return_expansion = rewards[t].unsqueeze(dim=-1) - self.alpha * log_probs[t].unsqueeze(dim=-1) + (1.0 - dones[t].unsqueeze(dim=-1)) * self.gamma * return_expansion

        mean_return_expansion = []

        for aidx in range(self.actor_iterations_per_epoch):
            avg_V = 0
            for bidx in range(self.actor_repeat_per_epoch):
                avg_V += return_expansion[aidx * self.actor_repeat_per_epoch + bidx]
            avg_V = avg_V / self.actor_repeat_per_epoch
            mean_return_expansion.append(avg_V)
        
        mean_return_expansion = torch.stack(mean_return_expansion, dim=0)

        return mean_return_expansion
"""
