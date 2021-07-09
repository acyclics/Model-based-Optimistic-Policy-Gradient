import torch
from mbopg.bh_mdp import Multi_SI_BayesNetwork

net = Multi_SI_BayesNetwork(2, 2, 10, [-1, -2], [4, 3])
w = net.sample(1)
obs = torch.ones((1, 2))
action = torch.ones((1, 2))
out1 = net.primarynet.batch_mbrl(obs, action, w)
out2 = net.primarynet(obs, action, w)
print("Obs1\n", out1)
print("Obs2", out2)
