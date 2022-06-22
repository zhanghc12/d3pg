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


class AutoregressiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_bin=40):
        super(AutoregressiveModel, self).__init__()
        self.input_dim = state_dim + action_dim + action_dim
        self.output_dim = num_bin
        self.l1 = nn.Linear(self.input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, self.output_dim)
        self.gap = 2. / num_bin

        self.zero_action = torch.zeros([256, action_dim]).to(device)
        self.one_hot_list = []
        self.action_masks = []
        for i in range(action_dim):
            self.one_hot_list.append(F.one_hot(torch.tensor([i]), num_classes=action_dim).repeat(256, 1).to(device))
            self.action_masks.append((torch.arange(action_dim) < i).float().repeat(256, 1).to(device))


    def forward(self, state, action):
        logits = []
        for i in range(action.shape[1]):
            action_mask = (torch.arange(action.shape[1]) < i).float().repeat(action.shape[0], 1).to(device)
            action_replaced = torch.where(action_mask > 0, action, self.zero_action)
            one_hot = self.one_hot_list[i]
            input = torch.cat([state.float(), action_replaced.float(), one_hot.float()], dim=1)
            logit = F.relu(self.l1(input))
            logit = F.relu(self.l2(logit))
            logit = self.l3(logit)
            logits.append(logit)
        return torch.cat(logits, dim=1)

    def get_logprob(self, state, action):
        logits = []
        label = ((action + 1) // self.gap).clamp_(0, self.output_dim-1).long().detach()

        for i in range(action.shape[1]):
            action_replaced = torch.where(self.action_masks[i].float() > 0, action.float(), self.zero_action.float())
            one_hot = self.one_hot_list[i]
            input = torch.cat([state.float(), action_replaced.float(), one_hot.float()], dim=1)
            logit = F.relu(self.l1(input))
            logit = F.relu(self.l2(logit))
            logit = self.l3(logit)
            logit = nn.LogSoftmax(dim=1)(logit)
            # logit = logit[label[:, i]]
            logit = logit.gather(dim=1, index=label[:,i:i+1])
            logit.unsqueeze(-1)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        logits = torch.sum(logits, dim=1)
        return logits

    def get_loss(self, state, action):
        loss = 0.
        label = ((action + 1) // self.gap).clamp_(0, self.output_dim-1).long().detach()

        for i in range(action.shape[1]):
            action_replaced = torch.where(self.action_masks[i].float() > 0, action.float(), self.zero_action.float())
            one_hot = self.one_hot_list[i]
            input = torch.cat([state.float(), action_replaced.float(), one_hot.float()], dim=1)
            logit = F.relu(self.l1(input))
            logit = F.relu(self.l2(logit))
            logit = self.l3(logit)
            loss += nn.CrossEntropyLoss()(logit, label[:, i])
        return loss



class TD3(object):
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale):
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

        self.auto_policy = AutoregressiveModel(state_dim, action_dim).to(device)
        self.auto_policy_optimizer = torch.optim.Adam(self.auto_policy.parameters(), lr=3e-4)

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


        # how to train the autoregressive model
        auto_policy_loss = self.auto_policy.get_loss(state, action)
        self.auto_policy_optimizer.zero_grad()
        auto_policy_loss.backward()
        self.auto_policy_optimizer.step()

        pi = self.actor(state)
        Q = self.critic.Q1(state, pi) + self.critic.Q2(state, pi)
        lmbda = 1 / Q.abs().mean().detach()
        logprob = self.auto_policy.get_logprob(state, pi)
        actor_loss = -lmbda * Q.mean() - self.bc_scale * logprob.mean()


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


