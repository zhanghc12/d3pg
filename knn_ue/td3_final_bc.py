import os
import torch
import torch.nn.functional as F
import copy
import numpy as np
import torch.nn as nn
from sf_offline.successor_feature import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAEPolicy(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim,
    ):
        super().__init__(
        )
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750)
        self.e2 = torch.nn.Linear(750, 750)

        self.mean = torch.nn.Linear(750, self.latent_dim)
        self.log_std = torch.nn.Linear(750, self.latent_dim)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 750)
        self.d2 = torch.nn.Linear(750, 750)
        self.d3 = torch.nn.Linear(750, action_dim)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).clamp(-0.5, 0.5).to(device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10, device='cpu'):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).clamp(-0.5,
                                                                                                                0.5).to(device)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_it = 0

        self.vae = VAEPolicy(state_dim, action_dim, action_dim * 2).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4)

    def train_vae(self, memory, batch_size):
        state, action, next_state, reward, not_done = memory.sample(batch_size)
        recon, mean, std = self.vae(state, action)
        recon_loss = nn.MSELoss()(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        self.total_it += 1

        return vae_loss.item(), recon_loss.item()

    def save_vae(self):


