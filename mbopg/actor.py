import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import mbopg.utils as utils


class DeterministicActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, noise_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.trunk_act = utils.mlp(obs_dim, hidden_dim, action_dim - noise_dim, hidden_depth)
        self.trunk_noise = utils.mlp(obs_dim, hidden_dim, noise_dim, hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs):
        act_mu = self.trunk_act(obs)
        noise_mu = self.trunk_noise(obs)

        act_mu = torch.tanh(act_mu)
        noise_mu = torch.tanh(noise_mu)

        return act_mu, noise_mu
