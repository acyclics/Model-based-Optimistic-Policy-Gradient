import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from mbopg.hyperfan import hyperfan_in_W_init, hyperfan_in_b_init, fanin_uniform


def sample_n(mu, sigma):
    eps = torch.randn_like(mu)
    z = mu + eps * sigma
    return z


def noise_generation(n_samples, noise_dim, device):
    #z = torch.randn((n_samples, noise_dim), device=device)
    z = (-2.0 - 2.0) * torch.rand((n_samples, noise_dim), device=device) + 2.0
    return z


class HypernetWeight(nn.Module):
    def __init__(self, shape, is_hyper_w, h_units=256, noise_shape=1, activation=nn.LeakyReLU()):
        super(HypernetWeight, self).__init__()
        self.shape = shape
        self.noise_shape = noise_shape
        self.act = activation

        bias = True
        noise_var = (2 + 2)**2 / 12.0
        #noise_var = 1.0
        n_noise = self.noise_shape
        relu = True
        out_bias = True

        self.f1 = nn.Linear(1, h_units, bias=bias)
        self.f2 = nn.Linear(h_units, h_units, bias=bias)

        self.f3 = nn.Linear(1, h_units, bias=bias)
        self.f4 = nn.Linear(h_units + h_units, h_units, bias=bias)

        self.f5 = nn.Linear(1, h_units, bias=bias)

        self.f6 = nn.Linear(h_units + h_units, h_units, bias=bias)
        self.f7 = nn.Linear(h_units, h_units, bias=bias)
        self.f8 = nn.Linear(h_units, h_units, bias=bias)
        self.f9 = nn.Linear(h_units, h_units, bias=bias)

        self.out = nn.Linear(h_units, shape, bias=bias)
        
        torch.nn.init.xavier_uniform_(self.f1.weight)
        torch.nn.init.zeros_(self.f1.bias)
        torch.nn.init.xavier_uniform_(self.f2.weight)
        torch.nn.init.zeros_(self.f2.bias)
        torch.nn.init.xavier_uniform_(self.f3.weight)
        torch.nn.init.zeros_(self.f3.bias)
        torch.nn.init.xavier_uniform_(self.f4.weight)
        torch.nn.init.zeros_(self.f4.bias)
        torch.nn.init.xavier_uniform_(self.f5.weight)
        torch.nn.init.zeros_(self.f5.bias)
        torch.nn.init.xavier_uniform_(self.f6.weight)
        torch.nn.init.zeros_(self.f6.bias)
        torch.nn.init.xavier_uniform_(self.f7.weight)
        torch.nn.init.zeros_(self.f7.bias)
        torch.nn.init.xavier_uniform_(self.f8.weight)
        torch.nn.init.zeros_(self.f8.bias)
        torch.nn.init.xavier_uniform_(self.f9.weight)
        torch.nn.init.zeros_(self.f9.bias)
       
        if is_hyper_w:
            hyperfan_in_W_init(self.out.weight, noise_var, n_noise, out_bias, relu)
            torch.nn.init.zeros_(self.out.bias)
        else:
            hyperfan_in_b_init(self.out.weight, noise_var, relu)
            torch.nn.init.zeros_(self.out.bias)
        
    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]

        w = self.f1(x1)
        w = self.act(w)
        w = self.f2(w)

        w2 = self.f3(x2)
        w = torch.cat([w, w2], dim=-1)
        w = self.act(w)

        w = self.f4(w)
        w3 = self.f5(x3)
        w = torch.cat([w, w3], dim=-1)
        w = self.act(w)

        w = self.f6(w)
        w = self.act(w)

        w = self.f7(w)
        w = self.act(w)

        w = self.f8(w)
        w = self.act(w)

        w = self.f9(w)
        w = self.act(w)

        w = self.out(w)

        return w.reshape((x.shape[0], self.shape))


class SingleHyperNetwork(nn.Module):

    def __init__(self, noise_dim, in_units, out_units, h_units, device):
        super(SingleHyperNetwork, self).__init__()
        self.noise_dim = noise_dim
        self.out_units = out_units
        self.in_units = in_units

        self.mu_w = HypernetWeight((out_units, in_units), True, h_units=h_units, noise_shape=noise_dim)
        self.mu_b = HypernetWeight((out_units, ), False, h_units=h_units, noise_shape=noise_dim)
    
    def forward(self, z):
        mu_w = self.mu_w(z)
        mu_b = self.mu_b(z)

        K = z.shape[0]

        mu_w = mu_w.view(K, -1)

        return mu_w


class LikelihoodNetwork(nn.Module):

    def __init__(self, noise_dim, obs_units, act_units, out_units, hidden_units, hidden_layers, obs_scale, device):
        super(LikelihoodNetwork, self).__init__()
        self.noise_dim = noise_dim
        self.obs_units = obs_units
        self.act_units = act_units
        self.out_units = out_units
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.obs_scale = obs_scale
        self.device = device

        self.mus = []

        self.obs_mu = nn.Parameter(torch.randn(hidden_units, obs_units), requires_grad=True)
        self.mus.append(self.obs_mu)
        self.obs_b = nn.Parameter(torch.zeros(hidden_units,), requires_grad=True)

        self.act_mu = nn.Parameter(torch.randn(hidden_units, act_units), requires_grad=True)
        self.mus.append(self.act_mu)
        self.act_b = nn.Parameter(torch.zeros(hidden_units,), requires_grad=True)

        self.tgt_mu = nn.Parameter(torch.randn(hidden_units, hidden_units + hidden_units), requires_grad=True)
        self.mus.append(self.tgt_mu)
        self.tgt_b = nn.Parameter(torch.zeros(hidden_units,), requires_grad=True)

        self.hidden_mu = torch.nn.ParameterList(parameters=[nn.Parameter(torch.randn(hidden_units, hidden_units), requires_grad=True) for _ in range(self.hidden_layers)])
        for i in range(hidden_layers):
            self.mus.append(self.hidden_mu[i])
        self.hidden_b = torch.nn.ParameterList(parameters=[nn.Parameter(torch.zeros(hidden_units,), requires_grad=True) for _ in range(self.hidden_layers)])

        self.state_mu = nn.Parameter(torch.randn(obs_units, hidden_units), requires_grad=True)
        self.mus.append(self.state_mu)
        self.state_b = nn.Parameter(torch.zeros(obs_units,), requires_grad=True)

        self.reward_mu = nn.Parameter(torch.randn(1, hidden_units), requires_grad=True)
        self.mus.append(self.reward_mu)
        self.reward_b = nn.Parameter(torch.zeros(1,), requires_grad=True)

        self.done_mu = nn.Parameter(torch.randn(1, hidden_units), requires_grad=True)
        self.mus.append(self.done_mu)
        self.done_b = nn.Parameter(torch.zeros(1,), requires_grad=True)


class PrimaryNetwork(nn.Module):

    def __init__(self, noise_dim, obs_units, act_units, out_units, hidden_units, hidden_layers, obs_scale, device, likelihoodnetwork):
        super(PrimaryNetwork, self).__init__()
        self.noise_dim = noise_dim
        self.obs_units = obs_units
        self.act_units = act_units
        self.out_units = out_units
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.obs_scale = obs_scale
        self.device = device
        self.llhnet = likelihoodnetwork

    def set_obs_trans(self, obs_min, obs_max):
        self.obs_min1 = np.expand_dims(obs_min, axis=0)
        self.obs_min2 = np.expand_dims(self.obs_min1, axis=0)
        self.obs_min1 = torch.from_numpy(self.obs_min1).to(self.device)
        self.obs_min2 = torch.from_numpy(self.obs_min2).to(self.device)

        self.obs_max1 = np.expand_dims(obs_max, axis=0)
        self.obs_max2 = np.expand_dims(self.obs_max1, axis=0)
        self.obs_max1 = torch.from_numpy(self.obs_max1).to(self.device)
        self.obs_max2 = torch.from_numpy(self.obs_max2).to(self.device)

    def feedforward(self, obs, action, w):
        w_obs_idx_s = 0
        w_obs_idx_e = self.obs_units * self.hidden_units
        b_idx = w_obs_idx_e

        w_obs = w[:, w_obs_idx_s : w_obs_idx_e]
        w_obs = w_obs.view(w.shape[0], self.hidden_units, self.obs_units)

        w_act_idx_s = b_idx
        w_act_idx_e = b_idx + self.act_units * self.hidden_units
        b_idx = w_act_idx_e

        w_act = w[:, w_act_idx_s : w_act_idx_e]
        w_act = w_act.view(w.shape[0], self.hidden_units, self.act_units)

        w_tgt_idx_s = b_idx
        w_tgt_idx_e = b_idx + self.hidden_units * self.hidden_units * 2
        b_idx = w_tgt_idx_e

        w_tgt = w[:, w_tgt_idx_s : w_tgt_idx_e]
        w_tgt = w_tgt.view(w.shape[0], self.hidden_units, self.hidden_units * 2)
        
        all_ws = []
        
        for i in range(self.hidden_layers):
            w_idx_s = b_idx
            w_idx_e = b_idx + self.hidden_units ** 2
            b_idx = w_idx_e

            w_h = w[:, w_idx_s : w_idx_e]
            w_h = w_h.view(w.shape[0], self.hidden_units, self.hidden_units)

            all_ws.append(w_h)
        
        w_state_idx_s = b_idx
        w_state_idx_e = b_idx + self.hidden_units * self.obs_units
        b_idx = w_state_idx_e

        w_state = w[:, w_state_idx_s : w_state_idx_e]
        w_state = w_state.view(w.shape[0], self.obs_units, self.hidden_units)

        w_reward_idx_s = b_idx
        w_reward_idx_e = b_idx + self.hidden_units * 1
        b_idx = w_reward_idx_e

        w_reward = w[:, w_reward_idx_s : w_reward_idx_e]
        w_reward = w_reward.view(w.shape[0], 1, self.hidden_units)

        w_done_idx_s = b_idx
        w_done_idx_e = b_idx + self.hidden_units * 1
        b_idx = w_done_idx_e

        w_done = w[:, w_done_idx_s : w_done_idx_e]
        w_done = w_done.view(w.shape[0], 1, self.hidden_units)

        next_states = []
        rewards = []
        dones = []

        #obs_unsqueezed = obs.unsqueeze(dim=1)

        for i in range(w.shape[0]):
            o = F.linear(obs, w_obs[i], self.llhnet.obs_b)
            a = F.linear(action, w_act[i], self.llhnet.act_b)

            y = torch.cat([o, a], dim=-1)
            y = F.leaky_relu(y)
            y = F.linear(y, w_tgt[i], self.llhnet.tgt_b)
            y = F.leaky_relu(y)

            for j in range(self.hidden_layers):
                y = F.linear(y, all_ws[j][i], self.llhnet.hidden_b[j])
                y = F.leaky_relu(y)
            
            next_state = F.linear(y, w_state[i], self.llhnet.state_b) + obs
            reward = F.linear(y, w_reward[i], self.llhnet.reward_b)
            done = F.linear(y, w_done[i], self.llhnet.done_b)

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        next_states = torch.stack(next_states, dim=1)

        rewards = torch.stack(rewards, dim=1)
        rewards = torch.tanh(rewards) * 2.0

        dones = torch.stack(dones, dim=1)
        dones = torch.sigmoid(dones)

        return next_states, rewards, dones

    def forward(self, obs, action, g):
        obs = (obs - self.obs_min1) / (self.obs_max1 - self.obs_min1)
        obs = (2 * obs - 1) * self.obs_scale
        next_states, rewards, dones = self.feedforward(obs, action, g)
        next_states = ((next_states / self.obs_scale) + 1) / 2
        next_states = next_states * (self.obs_max2 - self.obs_min2) + self.obs_min2
        return next_states, rewards, dones
    

class Multi_SI_BayesNetwork(nn.Module):

    def __init__(self, obs_units, act_units, obs_scale, obs_min, obs_max, hidden_units=64,
                 hidden_layers=3, weight_units=128, LLH_var=1.0, state_LLH_var=0.1, initial_logvar=-35.0,
                 lr=1e-4, grad_norm=5.0, noise_dim=3, entropy=1.0, device='cpu'):
        super(Multi_SI_BayesNetwork, self).__init__()
        self.LLH_var = LLH_var
        self.state_LLH_var = state_LLH_var

        self.device = device
        self.grad_norm = grad_norm

        self.K = 20
        self.J = 20
        self.noise_dim = noise_dim
        self.eps = 1e-10

        self.beta = 1.0
        self.entropy_term = entropy

        self.hidden_layers = hidden_layers

        self.obs_scale = obs_scale

        self.likelihoodnet = LikelihoodNetwork(self.noise_dim, obs_units, act_units, obs_units, hidden_units, hidden_layers, obs_scale, device)
        self.likelihoodnet = self.likelihoodnet.to(device)

        self.primarynet = PrimaryNetwork(self.noise_dim, obs_units, act_units, obs_units, hidden_units, hidden_layers, obs_scale, device, self.likelihoodnet)
        self.primarynet = self.primarynet.to(device)

        self.set_obs_trans(obs_min, obs_max)

        n_scales = hidden_units * (6 + self.hidden_layers)

        self.hypernet = HypernetWeight(n_scales, True, h_units=hidden_units, noise_shape=self.noise_dim)
        self.hypernet.to(device)

        self.params = list(self.hypernet.parameters()) + list(self.likelihoodnet.parameters())
        self.optim = torch.optim.Adam(self.params, lr=lr)

    def set_obs_trans(self, obs_min, obs_max):
        cloned_obs_max = obs_max.copy()
        cloned_obs_min = obs_min.copy()

        for idx in range(obs_min.shape[0]):
            if cloned_obs_max[idx] == cloned_obs_min[idx]:
                cloned_obs_min[idx] = 0.0
                cloned_obs_max[idx] = 1.0

        self.obs_min = np.expand_dims(cloned_obs_min, axis=0)
        self.obs_max = np.expand_dims(cloned_obs_max, axis=0)
        self.obs_min = torch.from_numpy(self.obs_min).to(self.device)
        self.obs_max = torch.from_numpy(self.obs_max).to(self.device)
        self.primarynet.set_obs_trans(cloned_obs_min, cloned_obs_max)

    def sample(self, num_samples=5, z=None):
        if z == None:
            z = noise_generation(num_samples, self.noise_dim, self.device)

        g = self.hypernet(z)
        logvar = torch.tensor([-2.3]).float().to(self.device)
        sigma = torch.exp(logvar / 2.0)
        gen_weights = sample_n(mu, sigma)

        return gen_weights

    def forward(self, obs, action, z):
        g = self.hypernet(z)

        mu, logvar = self.combinedhypernetwork.get_all_weights(num_samples)
        sigma = torch.exp(logvar / 2.0)
        w = sample_n(mu, sigma)
        y_pred = self.primarynet(x, w)
        #y_pred = y_pred.squeeze(dim=1)
        return y_pred

    def calculate_hypernet_loss(self, mu, sub_g_J, sub_g_K, logvar):
        mu = mu / torch.norm(mu, dim=1, keepdim=True)

        mu_J = mu.unsqueeze(dim=0).repeat(self.J, 1, 1) * sub_g_J.unsqueeze(dim=-1)
        mu_J = mu_J.view(mu_J.shape[0], -1)

        sigma = torch.exp(logvar / 2.0)
        w = sample_n(mu_J, sigma)

        mu_p = mu.unsqueeze(dim=0).repeat(self.J, 1, 1).detach() * sub_g_J.unsqueeze(dim=-1)
        mu_p = mu_p.view(mu_p.shape[0], -1)
        w_p = sample_n(mu_p, sigma)

        w_inner = w.unsqueeze(dim=1).repeat(1, self.K + 1, 1)

        mu = mu.detach()
        mu_star = mu.unsqueeze(dim=0).repeat(self.K, 1, 1) * sub_g_K.unsqueeze(dim=-1)
        mu_star = mu_star.view(mu_star.shape[0], -1)
        mu_star_inner = mu_star.unsqueeze(dim=0).repeat(self.J, 1, 1)
        mu_star_inner = torch.cat([mu_star_inner, mu_J.unsqueeze(dim=1)], dim=1)
        
        return w, w_p, w_inner, mu_star_inner

    def calculate_head_loss(self):
        z_J = noise_generation(self.J, self.noise_dim, self.device)
        z_K = noise_generation(self.K, self.noise_dim, self.device)

        g_J = self.hypernet(z_J)
        g_K = self.hypernet(z_K)

        w_inners = []
        mu_star_inners = []
        hypernet_ws = []
        hypernet_wps = []
        logvar = torch.tensor([-2.3]).float().to(self.device)#self.hypernet.logvar
        idx = 0

        for mu in self.likelihoodnet.mus:
            sub_g_J = g_J[:, idx:idx+mu.shape[0]]
            sub_g_K = g_K[:, idx:idx+mu.shape[0]]
            idx = idx + mu.shape[0]
            w, w_p, w_inner, mu_star_inner = self.calculate_hypernet_loss(mu, sub_g_J, sub_g_K, logvar)
            hypernet_ws.append(w)
            hypernet_wps.append(w_p)
            w_inners.append(w_inner)
            mu_star_inners.append(mu_star_inner)

        sigma = torch.exp(logvar / 2.0)
        hypernet_ws = torch.cat(hypernet_ws, dim=-1)
        hypernet_wps = torch.cat(hypernet_wps, dim=-1)
        w_inners = torch.cat(w_inners, dim=-1)
        mu_star_inners = torch.cat(mu_star_inners, dim=-1)

        # Entropy
        inner_sum = -0.5 * torch.sum(torch.pow(w_inners - mu_star_inners, 2) / (sigma ** 2), 2)
        log_H = torch.logsumexp(inner_sum, 1) - np.log(inner_sum.shape[1]) - 0.5 * logvar
        log_H = torch.mean(log_H)

        # Prior
        pi = 0.5
        sig1 = 1.0
        sig2 = 0.1
        log_prior1 = (-0.5 * torch.mean(torch.pow(hypernet_wps, 2) / sig1 ** 2)) - np.log(sig1 * np.sqrt(2 * np.pi))
        log_prior2 = (-0.5 * torch.mean(torch.pow(hypernet_wps, 2) / sig2 ** 2)) - np.log(sig2 * np.sqrt(2 * np.pi))
        log_prior = torch.stack([pi * log_prior1, (1.0 - pi) * log_prior2], dim=-1)
        log_prior = torch.logsumexp(log_prior, dim=-1)
                
        hypernet_loss = (log_prior - log_H * self.entropy_term) * self.beta

        return hypernet_loss, hypernet_ws

    def train(self, state, action, rewards, dones, next_states):
        state = (state - self.obs_min) / (self.obs_max - self.obs_min)
        state = (2 * state - 1) * self.obs_scale
        
        next_states = (next_states - self.obs_min) / (self.obs_max - self.obs_min)
        next_states = (2 * next_states - 1) * self.obs_scale

        rewards = rewards.unsqueeze(dim=1)
        next_states = next_states.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        hypernet_loss, hypernet_ws = self.calculate_head_loss()

        ss_mean, rs, ds = self.primarynet.feedforward(state, action, hypernet_ws)

        ss_log_lik = -0.5 * torch.sum((next_states - ss_mean) ** 2 / self.state_LLH_var, 2) - np.log(np.sqrt(self.state_LLH_var) * np.sqrt(2 * np.pi))
        rs_log_lik = -0.5 * torch.sum(torch.pow(rewards - rs, 2) / self.LLH_var, 2) - np.log(np.sqrt(self.LLH_var) * np.sqrt(2 * np.pi))
        ds_log_lik = torch.sum(dones * torch.log(ds + self.eps) + (1.0 - dones) * torch.log(1 - ds + self.eps), 2)

        log_lik = ss_log_lik + rs_log_lik + ds_log_lik
        log_lik = torch.mean(log_lik)
        loss = -(log_lik + hypernet_loss)
    
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm)
        self.optim.step()

        err1 = 0.5 * torch.sum((next_states - ss_mean) ** 2, 2)
        err2 = 0.5 * torch.sum(torch.pow(rewards - rs, 2), 2)
        err = 0.5 * (err1 + err2)
        err = torch.mean(err)
        
        return loss.detach(), torch.mean(log_lik).detach(), torch.mean(hypernet_loss).detach(), err.detach()
    
    def save(self, PATH):
        all_dict = dict()
        all_dict['combinedhypernetwork'] = self.combinedhypernetwork.get_state_dict()
        all_dict['optim'] = self.optim.state_dict()
        torch.save(all_dict, PATH)
    
    def load(self, PATH, cpu=False):
        if cpu:
            all_dict = torch.load(PATH, map_location=torch.device('cpu'))
        else:
            all_dict = torch.load(PATH, map_location=torch.device('cuda'))
        #self.optim.load_state_dict(all_dict['optim'])
        self.combinedhypernetwork.load(all_dict['combinedhypernetwork'])
    
    def get_state_dict(self, cpu=False):
        all_dict = dict()
        all_dict['primarynet'] = self.primarynet.state_dict()

        if cpu:
            all_dict['primarynet'] = {k: v.cpu() for k, v in all_dict['primarynet'].items()}

        all_dict['combinedhypernetwork'] = self.combinedhypernetwork.get_state_dict(cpu=cpu)

        #all_dict['optim_s'] = self.optim_s.state_dict()
        #all_dict['optim_r'] = self.optim_r.state_dict()
        #all_dict['optim_d'] = self.optim_d.state_dict()

        #if cpu:
        #    all_dict['optim_s'] = {k: v.cpu() for k, v in all_dict['optim_s'].items()}
        #    all_dict['optim_r'] = {k: v.cpu() for k, v in all_dict['optim_r'].items()} 
        #    all_dict['optim_d'] = {k: v.cpu() for k, v in all_dict['optim_d'].items()} 

        return all_dict
    
    def load_state_dict(self, all_dict):
        self.primarynet.load_state_dict(all_dict['primarynet'])
        self.combinedhypernetwork.load(all_dict['combinedhypernetwork'])

        #self.optim_s.load_state_dict(all_dict['optim_s'])
        #self.optim_r.load_state_dict(all_dict['optim_r'])
        #self.optim_d.load_state_dict(all_dict['optim_d'])


"""
class HypernetWeight(nn.Module):
    def __init__(self, shape, is_hyper_w, h_units=256, noise_shape=1, activation=nn.LeakyReLU()):
        super(HypernetWeight, self).__init__()
        self.shape = shape
        self.noise_shape = noise_shape
        self.act = activation

        bias = True
        noise_var = (2 + 2)**2 / 12.0
        #noise_var = 1.0
        n_noise = self.noise_shape
        relu = True
        out_bias = True

        self.f1 = nn.Linear(1, h_units, bias=bias)
        self.f2 = nn.Linear(h_units, h_units, bias=bias)

        self.f3 = nn.Linear(1, h_units, bias=bias)
        self.f4 = nn.Linear(h_units + h_units, h_units, bias=bias)

        self.f5 = nn.Linear(1, h_units, bias=bias)

        self.f6 = nn.Linear(h_units + h_units, h_units, bias=bias)
        self.f7 = nn.Linear(h_units, h_units, bias=bias)
        self.f8 = nn.Linear(h_units, h_units, bias=bias)
        self.f9 = nn.Linear(h_units, h_units, bias=bias)

        self.out = nn.Linear(h_units, np.prod(shape), bias=bias)
        
        torch.nn.init.xavier_uniform_(self.f1.weight)
        torch.nn.init.zeros_(self.f1.bias)
        torch.nn.init.xavier_uniform_(self.f2.weight)
        torch.nn.init.zeros_(self.f2.bias)
        torch.nn.init.xavier_uniform_(self.f3.weight)
        torch.nn.init.zeros_(self.f3.bias)
        torch.nn.init.xavier_uniform_(self.f4.weight)
        torch.nn.init.zeros_(self.f4.bias)
        torch.nn.init.xavier_uniform_(self.f5.weight)
        torch.nn.init.zeros_(self.f5.bias)
        torch.nn.init.xavier_uniform_(self.f6.weight)
        torch.nn.init.zeros_(self.f6.bias)
        torch.nn.init.xavier_uniform_(self.f7.weight)
        torch.nn.init.zeros_(self.f7.bias)
        torch.nn.init.xavier_uniform_(self.f8.weight)
        torch.nn.init.zeros_(self.f8.bias)
        torch.nn.init.xavier_uniform_(self.f9.weight)
        torch.nn.init.zeros_(self.f9.bias)
       
        if is_hyper_w:
            hyperfan_in_W_init(self.out.weight, noise_var, n_noise, out_bias, relu)
            torch.nn.init.zeros_(self.out.bias)
        else:
            hyperfan_in_b_init(self.out.weight, noise_var, relu)
            torch.nn.init.zeros_(self.out.bias)
        
    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]

        w = self.f1(x1)

        w = self.f2(w)
        w = self.act(w)

        w2 = self.f3(x2)

        w = torch.cat([w, w2], dim=-1)

        w = self.f4(w)
        w = self.act(w)

        w3 = self.f5(x3)

        w = torch.cat([w, w3], dim=-1)

        w = self.f6(w)
        w = self.act(w)

        w = self.f7(w)
        w = self.act(w)

        w = self.f8(w)
        w = self.act(w)

        w = self.f9(w)
        w = self.act(w)
        
        w = self.out(w)

        return w.reshape((x.shape[0], *self.shape))
"""
