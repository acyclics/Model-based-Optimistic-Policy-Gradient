import numpy as np


class ToyEnv():

    def __init__(self):
        self.obs_dim = 8
        self.act_dim = 8
        self.action_range = [0.01 for _ in range(self.act_dim)]
        self.action_mag = 0.01
    
    def initial_dataset(self):
        np.random.seed(0)

        n_dim = 8
        target_state = np.array([0.5, -0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5])

        dense_size = 950
        dense_state = np.random.uniform(0.0, 0.01, (dense_size, n_dim)).astype(np.float32)
        dense_action = []
        dense_rew = []
        dense_next_state = []
        dense_done = []
        for i in range(dense_size):
            act = np.random.uniform(-1.0, 1.0, (n_dim)) * self.action_mag
            s = dense_state[i]
            ns = s + act
            df = ns - target_state
            r = 0
            for j in range(n_dim):
                r += df[j]**2
            r = np.sqrt(r)
            dense_action.append(act)
            dense_next_state.append(ns)
            dense_rew.append(r)
            dense_done.append(False)

        inter_size = 50
        inter_state = []
        inter_action = []
        inter_rew = []
        inter_next_state = []
        inter_done = []
        s = np.array([1.0 for _ in range(n_dim)]).astype(np.float32)
        for i in range(inter_size):
            act = np.random.uniform(-1.0, 1.0, (n_dim)) * self.action_mag
            ns = s + act
            df = ns - target_state
            r = 0
            for j in range(n_dim):
                r += df[j]**2
            r = np.sqrt(r)
            inter_state.append(s)
            inter_action.append(act)
            inter_next_state.append(ns)
            inter_rew.append(r)
            inter_done.append(False)
            s = ns.copy()

        dataset_state = list(dense_state) + inter_state
        dataset_action = dense_action + inter_action
        dataset_rew = dense_rew + inter_rew
        dataset_next_state = dense_next_state + inter_next_state
        dataset_done = dense_done + inter_done
    
        return dataset_state, dataset_action, dataset_rew, dataset_next_state, dataset_done


if __name__ == '__main__':
    env = ToyEnv()
    dataset_state, dataset_action, dataset_rew, dataset_next_state, dataset_done = env.initial_dataset()
    print(dataset_state)



"""
class ToyEnv():

    def __init__(self):
        self.obs_dim = 3
        self.act_dim = 3
        self.action_range = [0.01, 0.01, 0.01]
        self.action_mag = 0.01
    
    def initial_dataset(self):
        np.random.seed(0)

        target_state = np.array([0.5, -0.5, 0.5])

        dense_size = 950
        dense_state = np.random.uniform(0.0, 0.01, (dense_size, 3)).astype(np.float32)
        dense_action = []
        dense_rew = []
        dense_next_state = []
        dense_done = []
        for i in range(dense_size):
            act = np.random.uniform(-1.0, 1.0, (3)) * self.action_mag
            s = dense_state[i]
            ns = s + act
            df = ns - target_state
            r = np.sqrt(df[0]**2 + df[1]**2 + df[2]**2) + np.sqrt(act[0]**2 + act[1]**2 + act[2]**2)
            dense_action.append(act)
            dense_next_state.append(ns)
            dense_rew.append(r)
            dense_done.append(False)

        inter_size = 50
        inter_state = []
        inter_action = []
        inter_rew = []
        inter_next_state = []
        inter_done = []
        s = np.array([1.0, 1.0, 1.0]).astype(np.float32)
        for i in range(inter_size):
            act = np.random.uniform(-1.0, 1.0, (3)) * self.action_mag
            ns = s + act
            df = ns - target_state
            r = np.sqrt(df[0]**2 + df[1]**2 + df[2]**2) + np.sqrt(act[0]**2 + act[1]**2 + act[2]**2)
            inter_state.append(s)
            inter_action.append(act)
            inter_next_state.append(ns)
            inter_rew.append(r)
            inter_done.append(False)
            s = ns.copy()

        dataset_state = list(dense_state) + inter_state
        dataset_action = dense_action + inter_action
        dataset_rew = dense_rew + inter_rew
        dataset_next_state = dense_next_state + inter_next_state
        dataset_done = dense_done + inter_done
    
        return dataset_state, dataset_action, dataset_rew, dataset_next_state, dataset_done
"""
