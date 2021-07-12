import numpy as np


class ToyEnv():

    def __init__(self):
        pass
    
    def initial_dataset(self):
        dense_size = 10000
        dense_state = np.random.uniform(-3.0, 3.0, (dense_size, 3)).astype(np.float32)
        dense_action = []
        dense_rew = []
        dense_next_state = []
        dense_done = []
        for i in range(dense_size):
            act = np.random.uniform(-0.1, 0.1, (3))
            s = dense_state[i]
            ns = s + act
            df = ns - s
            r = df[0] + df[1] + df[2]
            dense_action.append(act)
            dense_next_state.append(ns)
            dense_rew.append(r)
            dense_done.append(False)
        
        sparse

        return dense_state, dense_action, dense_rew, dense_next_state, dense_done


