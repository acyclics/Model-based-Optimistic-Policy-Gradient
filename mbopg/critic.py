import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import mbopg.utils as utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class DoubleVCritic(nn.Module):

    def __init__(self, obs_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.V1 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)
        self.V2 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        v1 = self.V1(obs)
        v2 = self.V2(obs)

        self.outputs['v1'] = v1
        self.outputs['v2'] = v2

        return v1, v2


class RetraceCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.A1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.V1 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)

        self.A2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.V2 = utils.mlp(obs_dim, hidden_dim, 1, hidden_depth)
    
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        a1 = self.A1(obs_action)
        a2 = self.A2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2
