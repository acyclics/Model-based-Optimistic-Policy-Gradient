import numpy as np
import torch


class ArtificialReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = torch.float32

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

    def initial_add(self, obs, action, reward, next_obs, done, done_no_max):
        idx = obs.shape[0]
        self.obses[0:idx] = obs.clone()
        self.actions[0:idx] = action.clone()
        self.rewards[0:idx] = reward.clone()
        self.next_obses[0:idx] = next_obs.clone()
        self.not_dones[0:idx] = 1.0 - done.clone()
        self.not_dones_no_max[0:idx] = 1.0 - done_no_max.clone()

        self.starting_idx = idx
        self.idx = idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        target_max_idx = self.idx + obs.shape[0]
        max_idx = min(target_max_idx, self.capacity)
        data_idx = target_max_idx - max_idx

        self.obses[self.idx:max_idx] = obs[data_idx:].clone()
        self.actions[self.idx:max_idx] = action[data_idx:].clone()
        self.rewards[self.idx:max_idx] = reward[data_idx:].clone()
        self.next_obses[self.idx:max_idx] = next_obs[data_idx:].clone()
        self.not_dones[self.idx:max_idx] = 1.0 - done[data_idx:].clone()
        self.not_dones_no_max[self.idx:max_idx] = 1.0 - done_no_max[data_idx:].clone()

        if data_idx != 0:
            self.idx = self.starting_idx
            self.full = True
        else:
            self.idx = target_max_idx

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
        