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

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2

class VAEPolicy():
    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim,
    ):
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750)
        self.e2 = torch.nn.Linear(750, 750)

        self.mean = torch.nn.Linear(750, self.latent_dim)
        self.log_std = torch.nn.Linear(750, self.latent_dim)

        self.d1 = torch.nn.Linear(self.latent_dim, 750)
        self.d2 = torch.nn.Linear(750, 750)
        self.d3 = torch.nn.Linear(750, obs_dim + action_dim)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.from_numpy(np.random.normal(0, 1, size=(std.size()))).to(device)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.from_numpy(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).clamp(-0.5, 0.5).to(device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))


    def decode_multiple(self, state, z=None, num_decode=10):
        if z is None:
            z = torch.from_numpy(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).clamp(-0.5,
                                                                                                                0.5).to(device)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)


class TD3(object):
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale, version):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = gamma
        self.tau = tau
        self.total_it = 0
        self.max_action = 1
        max_action = 1

        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.bc_scale = bc_scale

        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.log_beta = nn.Parameter(torch.zeros(1, requires_grad=True).to(device))
        self.beta_optimizer =torch.optim.Adam([self.log_beta], lr=3e-4)

        self.version = version

        '''
        self.mse_criterion = nn.MSELoss()
        self.vae = VAEPolicy(state_dim, action_dim, 2 * action_dim).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4)
        '''
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done, next_action, weights, idxes = memory.sample(batch_size)

        with torch.no_grad():
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

            # how to update the priority

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi = self.actor(state)
        Q = self.critic.Q1(state, pi) + self.critic.Q2(state, pi)
        lmbda = 1 / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean() + self.bc_scale * F.mse_loss(pi, action)

        #actor_loss = -lmbda * Q.mean() + (self.bc_scale * torch.abs(self.critic.Q1(state, pi) - self.critic.Q2(state, pi))).mean()

        #actor_loss = Q.mean() + (self.bc_scale * torch.abs(self.critic.Q1(state, pi) - self.critic.Q2(state, pi))).mean()
        #actor_loss = (self.bc_scale * torch.abs(self.critic.Q1(state, pi) - self.critic.Q2(state, pi))).mean()

        if self.version == 2:
            uncertainty = torch.clamp(torch.abs((self.critic.Q1(state, pi) - self.critic.Q2(state, pi)) / (self.critic.Q1(state, pi) + self.critic.Q2(state, pi) + 1e-2) ), 0 ,1)
            uncertainty_loss = self.log_beta.exp() * (uncertainty - self.bc_scale).mean()
            actor_loss = -lmbda * Q.mean() + uncertainty_loss
            if self.total_it % self.policy_freq == 0:
                self.beta_optimizer.zero_grad()
                (-uncertainty_loss).backward(retain_graph=True)
                self.beta_optimizer.step()
                self.log_beta.data.clamp_(min=-5.0, max=10.0)

        # actor_loss = - Q.mean()

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

        priority = 2 - (torch.sqrt(torch.mean((self.actor(next_state) - next_action) ** 2, dim=1))) + 1e-3
        if self.version == 3:
            priority = torch.exp(- (torch.mean((self.actor(next_state) - next_action) ** 2, dim=1)))
        priority = priority.detach().cpu().numpy()
        memory.update_priorities(idxes, priority)

        '''
        recon, mean, std = self.vae(state, action)
        recon_loss = self.mse_criterion(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        '''

        return critic_loss.item(), actor_loss.item()


