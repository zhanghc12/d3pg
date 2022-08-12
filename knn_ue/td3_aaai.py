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


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module('fc' + str(i), fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output

class VAE(nn.Module):
    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim=6,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750)
        self.e2 = torch.nn.Linear(750, 750)

        self.mean = torch.nn.Linear(750, self.latent_dim)
        self.log_std = torch.nn.Linear(750, self.latent_dim)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 750)
        self.d2 = torch.nn.Linear(750, 750)
        self.d3 = torch.nn.Linear(750, obs_dim + action_dim)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def forward_encoder_decoder(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.from_numpy(np.random.normal(0, 1, size=(std.size()))).float().to(device)
        u = self.decode(state, z)

        return u, mean, std

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        return mean

    def decode(self, state, z=None):
        if z is None:
            z = torch.from_numpy(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).clamp(-0.5, 0.5).float().to(device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [512, 512, 512], n_quantiles)
            self.add_module('qf' + str(i), net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__(validate_args=False)
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device),
                                      torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + F.logsigmoid(2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh


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
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale, eta, n_nets, n_quantiles, top_quantiles_to_drop=200, drop_quantile_bc=0, output_dim=9, version=0, k=1, is_random=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = gamma
        self.tau = tau
        self.bc_scale = bc_scale
        self.total_it = 0

        n_nets = n_nets
        n_quantiles = n_quantiles

        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.actor = Actor(state_dim, action_dim, 1).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, n_quantiles, n_nets).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.target_entropy = - action_dim

        self.quantiles_total = n_quantiles * n_nets
        self.base_tensor = torch.ones([256, 1]).to(device)
        self.mask = torch.arange(self.quantiles_total).repeat(256, 1).to(device) # batch * totoal_quantile
        self.state_dim = state_dim
        self.action_dim = action_dim


        self.drop_quantile_bc = drop_quantile_bc
        #upper_bound = 0.01
        #lower_bound = 0.003

        upper_bound = 0.05
        lower_bound = 0.02
        self.scale = 1 / (upper_bound - lower_bound)
        self.bias = - lower_bound * self.scale
        self.eta = eta
        self.version = version
        self.k = k
        self.is_random = is_random
        if is_random == 0:
            self.feature_nn = VAE(state_dim, action_dim, output_dim).to(device)
            self.feature_criterion = nn.MSELoss()
            self.feature_optimizer = torch.optim.Adam(self.feature_nn.parameters(), lr=3e-4)

        else:
            self.feature_nn = FeatureExtractorV4(state_dim, action_dim, 256, output_dim).to(device)

    def train_feature_extractor(self, memory, batch_size):
        assert self.is_random == 0
        state, action, _, _, _ = memory.sample(batch_size)
        recon, mean, std = self.feature_nn.forward_encoder_decoder(state, action)
        recon_loss = self.feature_criterion(recon, torch.cat([state, action], dim=1))
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.feature_optimizer.zero_grad()
        vae_loss.backward()
        self.feature_optimizer.step()

    def select_action(self, state, evaluate=False, bc=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def get_stat(self, mean_distance):
        self.mean_distance = mean_distance


    def train_policy(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)
            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()

            target_distance = kd_trees.query(query_data, k=self.k)[0] # / (self.state_dim + self.action_dim)
            target_distance = np.mean(target_distance, axis=1, keepdims=True)
            cond = -torch.clamp_(self.eta * torch.FloatTensor(target_distance).to(self.device), 0, 1) * 100 + 150

        mask = (self.mask < cond).float()  # batch * total_quantile
        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))

        # --- Policy and alpha loss --
        if self.total_it % 2 == 0:

            if self.version == 1:
                query_data = self.feature_nn(state, self.actor(state)).detach().cpu().numpy()
                target_distance = kd_trees.query(query_data, k=self.k)[0]  # / (self.state_dim + self.action_dim)
                target_distance = np.mean(target_distance, axis=1, keepdims=True)

                actor_scale = torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1)
                actor_loss = ((1 - actor_scale) * source_loss).mean() + (actor_scale * bc_loss).mean()
            elif self.version == 3:
                actor_loss = source_loss.mean()
            elif self.version == 9:
                actor_loss = (actor_loss + bc_loss).mean()
            else:
                raise NotImplementedError
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), 0.


def quantile_huber_loss_f(quantiles, samples, mask=None):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples

    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    if mask is None:
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    else:
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss)
        loss = loss * mask[:, None, None, :]
        loss = loss.mean()

    return loss

