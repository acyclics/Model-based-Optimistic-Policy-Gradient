import torch
import numpy as np
from mbopg.bh_mdp import Multi_SI_BayesNetwork

net = Multi_SI_BayesNetwork(2, 2, 10, np.array([-1, -2]), np.array([4, 3]))
bs = 5
w = net.sample(bs)
obs = torch.ones((bs, 2))
action = torch.ones((bs, 2))
out1 = net.primarynet.batch_mbrl(obs, action, w)
out2 = []
for idx in range(bs):
    o2 = net.primarynet(obs[idx:idx+1], action[idx:idx+1], w[idx:idx+1])
    out2.append(o2)
print("Obs1\n", out1)
print("Obs2", out2)
