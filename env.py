import numpy as np


class EnvWrapper():

    def __init__(self, env, stack_steps=1):
        self.env = env
        self.stack_steps = stack_steps
    
    def return_stacked_obs(self):
        flattened_obs = np.concatenate(past_obses, axis=0)
        return flattened_obs

    def append_stacked_obs(self, new_obs):
        self.past_obses = self.past_obses[1:]
        self.past_obses.append(new_obs)

    def reset(self):
        obs = self.env.reset()
        self.past_obses = [np.zeros(self.env.observation_space.shape[0]) for _ in range(self.stack_steps - 1)]
        self.past_obses.append(obs)
        return self.return_stacked_obs()
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space * self.stack_steps

    def step(self, action):
        next_state, rew, done, info = self.env.step(action)
        self.append_stacked_obs(next_state)
        return self.return_stacked_obs(), rew, done, info
    
    def close(self):
        self.env.close()
