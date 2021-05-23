import numpy as np
import torch


class ArtificialReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = torch.float32 if len(obs_shape) == 1 else torch.uint8

        self.obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype, device=device)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype, device=device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.not_dones_no_max = torch.empty((capacity, 1), dtype=torch.float32, device=device)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.obses[self.idx] = obs.clone()
        self.actions[self.idx] = action.clone()
        self.rewards[self.idx] = reward.clone()
        self.next_obses[self.idx] = next_obs.clone()
        self.not_dones[self.idx] = not done
        self.not_dones_no_max[self.idx] = not done_no_max

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs].clone().float()
        actions = self.actions[idxs].clone()
        rewards = self.rewards[idxs].clone()
        next_obses = self.next_obses[idxs].clone().float()
        not_dones = self.not_dones[idxs].clone()
        not_dones_no_max = self.not_dones_no_max[idxs].clone()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
        