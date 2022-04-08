import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from noisy_ensemble.noisy_network import QNoisyNetwork# , GaussianPolicy

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


class NoisyNetoworkEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, n_nets):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = QNoisyNetwork(state_dim, action_dim, 256)
            self.add_module('qf' + str(i), net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        outputs = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return outputs


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [256, 256], 2 * action_dim)

    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(device)[None, :]
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action

    def log_prob(self, obs, action):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)
        y_t = action
        x_t = atanh(y_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return log_prob


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
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


class NEO(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, top_quantiles_to_drop_per_net=2, n_nets=5, bc_scale=0.2):
        n_quantiles = 25
        n_nets = n_nets
        target_entropy = - action_dim
        top_quantiles_to_drop = top_quantiles_to_drop_per_net

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = NoisyNetoworkEnsemble(state_dim, action_dim, n_nets).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.version = version

        self.target_threshold = target_threshold # note:
        self.target_entropy = target_entropy
        self.total_it = 0

        self.log_alpha = torch.zeros((1,), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.n_nets = n_nets
        self.bc_scale = bc_scale

    def select_action(self, state):
        return self.actor.select_action(state)

    def train_without_bc(self, replay_buffer, batch_size=256):
        self.total_it += 1
        alpha = torch.exp(self.log_alpha)
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_Q = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            next_Q = torch.min(next_Q, dim=0)
            # compute target
            target = reward + not_done * self.discount * (next_Q - alpha * next_log_pi) # batch_size * 1

        cur_Q = self.critic(state, action) # n * batch_size * 1
        # critic_loss = ensemble_critic_loss(cur_Q, target)
        critic_loss = ((cur_Q - target.unsqueeze(0)) ** 2).mean()

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.critic(state, new_action).min(dim=0)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.total_it += 1
        return actor_loss.item(), critic_loss.item(), 0, 0, 0

    def train_with_bc(self, replay_buffer, batch_size=256):
        self.total_it += 1
        alpha = torch.exp(self.log_alpha)
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_Q = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            next_Q = torch.min(next_Q, dim=0)
            # compute target
            target = reward + not_done * self.discount * (next_Q - alpha * next_log_pi) # batch_size * 1

        cur_Q = self.critic(state, action) # n * batch_size * 1
        # critic_loss = ensemble_critic_loss(cur_Q, target)
        critic_loss = ((cur_Q - target.unsqueeze(0)) ** 2).mean()

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        pi_value = self.critic(state, new_action).min(dim=0)
        # actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
        actor_loss = (alpha * log_pi - pi_value)
        lmbda = 2.5 / actor_loss.abs().mean().detach()
        actor_loss = actor_loss.mean()

        if self.version == 1:
            true_actor_loss = actor_loss
            behavior_log_prob = self.actor.log_prob(state, action)
            bc_loss = self.bc_scale * behavior_log_prob.mean()

            actor_loss = actor_loss * lmbda - self.bc_scale * behavior_log_prob.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.total_it += 1

        return actor_loss.item(), critic_loss.item()


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

