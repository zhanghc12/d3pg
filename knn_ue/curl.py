import os
import torch
import torch.nn.functional as F
import copy
import numpy as np
import torch.nn as nn
from torch.distributions import Distribution, Normal
from sf_offline.successor_feature import Actor
from tqc.spectral_normalization import spectral_norm


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

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2



class ResBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h

class FeatureExtractorV4(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, feat_dim=3):
        super(FeatureExtractorV4, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim
        norm_bound = 10
        n_power_iterations = 1

        # first layer feature

        self.feature_l1 = nn.Sequential(

            nn.Linear(self.state_dim + self.action_dim, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )

        self.feature_l2 = nn.Linear(hidden_dim, self.feat_dim)

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        w = F.relu(self.feature_l1(input))
        w = self.feature_l2(w)
        return w



class TD3(object):
    def __init__(self, state_dim, action_dim, gamma, tau, target_threshold, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = gamma
        self.tau = tau
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

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.target_threshold = target_threshold
        self.feature_nn = FeatureExtractorV4(state_dim, action_dim, 256, output_dim).to(device)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def get_stat(self, mean_distance):
        self.mean_distance = mean_distance

    def train(self, memory, batch_size, kd_trees):
        self.total_it += 1

        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # Compute the target Q value
            next_action = self.actor_target(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

            # how to update the priority

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        query_data = self.feature_nn(next_state, next_action).detach().cpu().numpy()
        target_distance = kd_trees.query(query_data, k=1)[0] # / 20

        target_distance = torch.FloatTensor(target_distance).to(self.device)
        #tree_index = np.random.choice(len(kd_trees))
        #kd_tree = kd_trees[tree_index]
        #target_distance = kd_tree.query(query_data, k=1)[0] / (self.state_dim + self.action_dim)

        # actor_scale = torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1)
        next_Q1, next_Q2 = self.critic(next_state, next_action)
        next_Q = (next_Q1 + next_Q2) / 2
        curl_loss = (next_Q * (target_distance > self.target_threshold).float()).mean()

        critic_loss = critic_loss + curl_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi = self.actor(state)
        Q = self.critic.Q1(state, pi) + self.critic.Q2(state, pi)
        # lmbda = self.alpha / Q.abs().mean().detach()
        # actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

        actor_loss = - Q.mean()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), curl_loss.item()



