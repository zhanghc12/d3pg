import os
import torch
import torch.nn.functional as F
import copy
import numpy as np
import torch.nn as nn
from torch.distributions import Distribution, Normal
from sf_offline.successor_feature import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN_MAX = (-20, 2)

epsilon = 1e-6


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

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class UncertaintyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(UncertaintyNet, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        u = F.relu(self.l1(sa))
        u = F.relu(self.l2(u))
        u = self.l3(u)
        return u


class TD3(object):
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale, n_nets, n_quantiles, version=0, top_quantiles_to_drop=200, drop_quantile_bc=0, output_dim=9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = gamma
        self.tau = tau
        self.bc_scale = bc_scale
        self.total_it = 0
        self.max_action = 1
        max_action = 1

        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.alpha = 2.5

        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.num_critic = 8 # num_critic
        self.critics = nn.ModuleList()
        for i in range(self.num_critic):
            self.critics.append(Critic(state_dim, action_dim))
        self.critics.to(device)
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)
        self.critics_target = copy.deepcopy(self.critics)

        # define the uncertainty estimator
        self.uncertainty_net = UncertaintyNet(state_dim, action_dim).to(device)
        self.uncertainty_optimizer = torch.optim.Adam(self.uncertainty_net.parameters(), lr=3e-4)

        self.version = version

        if self.version == 2:
            self.target_uncertainty = 0.1
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)


    def select_action(self, state, evaluate=False, bc=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def get_stat(self, mean_distance):
        self.mean_distance = mean_distance


    def train(self, memory, batch_size, kd_trees):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        critics_loss = 0.
        next_action = self.actor_target(next_state).detach()
        next_uncertainty_list = []
        # Compute the target Q value
        for critic, critic_target in zip(self.critics, self.critics_target):
            with torch.no_grad():
                target_Q1, target_Q2 = critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q
                next_uncertainty_list.append((target_Q1 + target_Q2)/2)

            # Get current Q estimates
            current_Q1, current_Q2 = critic(state, action)

            # Compute critic loss
            critics_loss += F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        self.critics_optimizer.step()

        # todo: rescale the uncertainty things, Q is not comparable, for all the Q?
        next_uncertainty = torch.cat(next_uncertainty_list, dim=1)
        next_uncertainty = torch.clamp(torch.std(next_uncertainty, dim=1) / (torch.mean(torch.abs(next_uncertainty), dim=1) + 1e-3), 0, 1)
        cur_uncertainty = self.uncertainty_net(state, action)
        target_uncertainty = next_uncertainty + not_done * self.discount * self.uncertainty_net(next_state, next_action)
        uncertainty_loss = F.mse_loss(cur_uncertainty, target_uncertainty.detach())
        self.uncertainty_optimizer.zero_grad()
        uncertainty_loss.backward()
        self.uncertainty_optimizer.step()

        actor_loss = 0.
        Qs = []
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            for critic in self.critics:
                Qs.append((critic.Q1(state, pi) + critic.Q2(state, pi)) / 2)
            Q = torch.cat(Qs, dim=1)
            lmbda = self.alpha / Q.abs().mean().detach()
            pi_uncertainty = self.uncertainty_net(state, pi)

            # how to incorporate uncertainty into the mse_loss
            if self.version == 0:
                actor_loss = -lmbda * Q.mean() + ((pi - action) * (pi - action)).mean()
            elif self.version == 1:
                actor_loss = -lmbda * Q.mean() + (pi_uncertainty.detach() * (pi - action) * (pi - action)).mean()
            elif self.version == 2:
                actor_loss = -lmbda * Q.mean() + self.log_alpha.exp() * pi_uncertainty.mean()
                # Optimize the alpha
                alpha_loss = -(self.log_alpha.exp() * (pi_uncertainty - self.target_uncertainty).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward(retain_graph=True)
                self.alpha_optim.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critics_loss.item(), actor_loss


