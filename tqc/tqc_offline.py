import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal


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


class TQC(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, top_quantiles_to_drop_per_net=2, n_nets=5, bc_scale=0.2):
        n_quantiles = 25
        n_nets = n_nets
        target_entropy = - action_dim
        top_quantiles_to_drop = top_quantiles_to_drop_per_net

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim, n_quantiles, n_nets).to(device)
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

        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.quantiles_total = n_quantiles * n_nets
        self.normalized_std_z_ood = 0
        self.normalized_std_z_iod = 0
        self.n_nets = n_nets
        self.n_quantiles = n_quantiles
        self.base_tensor = torch.ones([256, 1]).to(device)
        self.mask = torch.arange(self.quantiles_total).repeat(256, 1).to(device) # batch * totoal_quantile
        self.remained_quantiles = self.quantiles_total - self.top_quantiles_to_drop
        self.bc_scale = bc_scale

    def select_action(self, state):
        return self.actor.select_action(state)

    def train_without_mask(self, replay_buffer, batch_size=256):
        self.total_it += 1
        alpha = torch.exp(self.log_alpha)
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))  # batch * 250
            sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop] # batch * 200
            # compute target
            target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.total_it += 1


        cur_z = self.critic_target(state, action)
        std_z_iid = torch.std(cur_z, dim=1, keepdim=False)
        std_z_iid = std_z_iid.mean()
        normalized_std_z_iid = std_z_iid / cur_z.mean()

        self.normalized_std_z_iod = self.normalized_std_z_iod * 0.995 + 0.005 * normalized_std_z_iid.item()

        std_z_ood = torch.std(next_z, dim=1, keepdim=False)
        std_z_ood = std_z_ood.mean()
        normalized_std_z_ood = std_z_ood / next_z.mean()
        self.normalized_std_z_ood = self.normalized_std_z_ood * 0.9 + 0.1 * normalized_std_z_ood.item()

        if self.normalized_std_z_ood > self.normalized_std_z_iod * 1.1:
            self.top_quantiles_to_drop = int(max(min(self.top_quantiles_to_drop * 1.1, self.quantiles_total - 10), 50))
        else:
            self.top_quantiles_to_drop = int(max(min(self.top_quantiles_to_drop * 0.9, self.quantiles_total - 10), 50))

        return actor_loss.item(), critic_loss.item(), self.top_quantiles_to_drop, self.normalized_std_z_iod, self.normalized_std_z_ood

    def train_mask(self, replay_buffer, batch_size=256):
        self.total_it += 1
        alpha = torch.exp(self.log_alpha)
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            # sorted_z = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
            # compute target
            target = reward + not_done * self.discount * (sorted_z - alpha * next_log_pi)

        cur_z = self.critic(state, action)
        # define an mask matrix

        cur_tar_z = self.critic_target(state, action)  # batch * nets * quantiles
        std_z_iid = torch.std(cur_tar_z, dim=1, keepdim=False)  # batch * quantiles
        std_z_iid = std_z_iid / (torch.abs(cur_tar_z).mean(dim=1, keepdim=False) + 1e-2) # batch * quantiles
        normalized_std_z_iid = std_z_iid.mean()
        self.normalized_std_z_iod = self.normalized_std_z_iod * 0.995 + 0.005 * normalized_std_z_iid.item()

        std_z_ood = torch.std(next_z, dim=1, keepdim=False)  # batch * quantiles
        std_z_ood = std_z_ood / (torch.abs(next_z).mean(dim=1, keepdim=False) + 1e-2)  # batch * quantile
        std_z_ood = std_z_ood.mean(dim=1, keepdim=True)  # batch * 1
        cond = torch.where(std_z_ood - self.normalized_std_z_iod * 2 > 0, 0.04 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        cond = torch.where(std_z_ood - self.normalized_std_z_iod * 1.1 < 0, 0.6 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        # version rl1 : x > 1.1, 10 |  x < 1.1, 150
        # version rl3 : x > 2, 10,  | 1.1 < x < 2, 50, | x < 1.1, 150
        mask = (self.mask < cond).float()  # batch * total_quantile

        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.total_it += 1


        return actor_loss.item(), critic_loss.item(), self.top_quantiles_to_drop, self.normalized_std_z_iod, std_z_ood.mean().item()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        alpha = torch.exp(self.log_alpha)
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            # sorted_z = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
            # compute target
            target = reward + not_done * self.discount * (sorted_z - alpha * next_log_pi)

        cur_z = self.critic(state, action)
        # define an mask matrix
        '''
        cur_tar_z = self.critic_target(state, action)  # batch * nets * quantiles
        std_z_iid = torch.std(cur_tar_z, dim=1, keepdim=False)  # batch * quantiles
        std_z_iid = std_z_iid / (torch.abs(cur_tar_z).mean(dim=1, keepdim=False) + 1e-2) # batch * quantiles
        normalized_std_z_iid = std_z_iid.mean()
        self.normalized_std_z_iod = self.normalized_std_z_iod * 0.995 + 0.005 * normalized_std_z_iid.item()

        std_z_ood = torch.std(next_z, dim=1, keepdim=False)  # batch * quantiles
        std_z_ood = std_z_ood / (torch.abs(next_z).mean(dim=1, keepdim=False) + 1e-2)  # batch * quantile
        std_z_ood = std_z_ood.mean(dim=1, keepdim=True)  # batch * 1
        cond = torch.where(std_z_ood - self.normalized_std_z_iod * 2 > 0, 0.04 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        cond = torch.where(std_z_ood - self.normalized_std_z_iod * 1.1 < 0, 0.6 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        # version rl1 : x > 1.1, 10 |  x < 1.1, 150
        # version rl3 : x > 2, 10,  | 1.1 < x < 2, 50, | x < 1.1, 150
        mask = (self.mask < cond).float()  # batch * total_quantile
        '''
        critic_loss = quantile_huber_loss_f(cur_z, target)


        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        # the core idea is the actor things, actor must stay in the distribution,

        pi_value = self.critic(state, new_action)  # batch x nets x quantiles
        sorted_z, _ = torch.sort(pi_value.reshape(batch_size, -1))
        sorted_z = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
        pi_value = sorted_z.mean(1, keepdim=True)
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

        return actor_loss.item(), critic_loss.item(), self.top_quantiles_to_drop, self.normalized_std_z_iod, self.normalized_std_z_iod, true_actor_loss.item(), bc_loss.item()



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