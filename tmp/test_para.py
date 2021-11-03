import torch

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 2)
        self.l2 = nn.Linear(2, 2)
        self.l3 = nn.Linear(2, 1)
        self.max_logvar = nn.Parameter((torch.ones((1, state_dim)).float() / 2), requires_grad=True)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q), self.max_logvar


critic = Critic(10, 5).to(device)
a,b = critic(torch.normal(torch.zeros([10,10]), torch.ones([10,10])), torch.normal(torch.zeros([10,5]), torch.ones([10,5])))

optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = a.mean() + 0.01 * b.sum()
loss.backward()

optimizer.step()
print(critic.max_logvar)
# print(list(critic.parameters()))
