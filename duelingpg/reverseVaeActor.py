import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RerverseVaeActor(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750)
        self.e2 = torch.nn.Linear(750, 750)

        self.mean = torch.nn.Linear(750, self.latent_dim)
        self.log_std = torch.nn.Linear(750, self.latent_dim)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 750)
        self.d2 = torch.nn.Linear(750, 750)
        self.d3 = torch.nn.Linear(750, action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, next_state):
        return self.decode(next_state), 0

    def forward(self, next_state, action):
        z = F.relu(self.e1(torch.cat([next_state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.from_numpy(np.random.normal(0, 1, size=(std.size()))).float().to(self.device)

        u = self.decode(next_state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.from_numpy(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).float().clamp(-0.5, 0.5).to(self.device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.d3(a)