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
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale, n_nets, n_quantiles, top_quantiles_to_drop=200, drop_quantile_bc=0):
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

        self.feature_nn = FeatureExtractorV4(state_dim, action_dim, 256, 9).to(device)

        self.drop_quantile_bc = drop_quantile_bc
        #upper_bound = 0.01
        #lower_bound = 0.003

        upper_bound = 0.05
        lower_bound = 0.02
        self.scale = 1 / (upper_bound - lower_bound)
        self.bias = - lower_bound * self.scale

    def select_action(self, state, evaluate=False, bc=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def get_stat(self, mean_distance):
        self.mean_distance = mean_distance


    def train_policy_quantile_without_uncertainty(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
            target = reward + not_done * self.discount * (sorted_z)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        actor_loss = (- self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)).mean()
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()

    def train_policy_quantile_with_uncertainty(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            #next_state_batch_np = next_state.cpu().numpy()
            #next_action_batch_np = new_next_action.detach().cpu().numpy()
            #query_data = np.concatenate([next_state_batch_np, next_action_batch_np], axis=1)
            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()
            # query_data = torch.cat([next_state, new_next_action], dim=1).detach().cpu().numpy()

            #tree_index = np.random.choice(len(kd_trees))
            # kd_tree = kd_trees[tree_index]

            # target_distance = kd_tree.query(query_data, k=1)[0]
            # target_distance = torch.FloatTensor(target_distance).to(self.device)

            target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)
            cond = -torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1) * 100 + 150

            # target_flag = torch.FloatTensor(target_distance < self.mean_distance).to(self.device)

        # Get current Q estimates

        # mask calculation
        # cond = torch.where(target_distance - 0.25 > 0, 0.0 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        # cond = torch.where(target_distance - 0.15 < 0, 0.8 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        mask = (self.mask < cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        actor_loss = (- self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)).mean()
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()


    def train_policy_quantile_with_uncertainty_v2(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            #next_state_batch_np = next_state.cpu().numpy()
            #next_action_batch_np = new_next_action.detach().cpu().numpy()
            #query_data = np.concatenate([next_state_batch_np, next_action_batch_np], axis=1)
            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()
            # query_data = torch.cat([next_state, new_next_action], dim=1).detach().cpu().numpy()

            #tree_index = np.random.choice(len(kd_trees))
            # kd_tree = kd_trees[tree_index]

            # target_distance = kd_tree.query(query_data, k=1)[0]
            # target_distance = torch.FloatTensor(target_distance).to(self.device)

            target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)
            cond = -torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1) * 130 + 150
            # cond = -torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1) * 230 + 230

            # target_flag = torch.FloatTensor(target_distance < self.mean_distance).to(self.device)

        # Get current Q estimates

        # mask calculation
        # cond = torch.where(target_distance - 0.25 > 0, 0.0 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        # cond = torch.where(target_distance - 0.15 < 0, 0.8 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        mask = (self.mask < cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        actor_loss = (- self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)).mean()
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()


    def train_policy_quantile_with_uncertainty_v3(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            #next_state_batch_np = next_state.cpu().numpy()
            #next_action_batch_np = new_next_action.detach().cpu().numpy()
            #query_data = np.concatenate([next_state_batch_np, next_action_batch_np], axis=1)
            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()
            # query_data = torch.cat([next_state, new_next_action], dim=1).detach().cpu().numpy()

            #tree_index = np.random.choice(len(kd_trees))
            # kd_tree = kd_trees[tree_index]

            # target_distance = kd_tree.query(query_data, k=1)[0]
            # target_distance = torch.FloatTensor(target_distance).to(self.device)

            target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)
            actor_scale = torch.clamp_(self.scale * torch.FloatTensor(target_distance).to(self.device) + self.bias, 0, 1)

            cond = -actor_scale * self.bc_scale + self.bc_scale

            # target_flag = torch.FloatTensor(target_distance < self.mean_distance).to(self.device)

        # Get current Q estimates

        # mask calculation
        # cond = torch.where(target_distance - 0.25 > 0, 0.0 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        # cond = torch.where(target_distance - 0.15 < 0, 0.8 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        mask = (self.mask <= cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        actor_loss = (- self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)).mean()
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()



    def train_policy_quantile_with_uncertainty_v1(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            #next_state_batch_np = next_state.cpu().numpy()
            #next_action_batch_np = new_next_action.detach().cpu().numpy()
            #query_data = np.concatenate([next_state_batch_np, next_action_batch_np], axis=1)
            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()
            # query_data = torch.cat([next_state, new_next_action], dim=1).detach().cpu().numpy()

            #tree_index = np.random.choice(len(kd_trees))
            # kd_tree = kd_trees[tree_index]

            # target_distance = kd_tree.query(query_data, k=1)[0]
            # target_distance = torch.FloatTensor(target_distance).to(self.device)

            target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)
            actor_scale = torch.clamp_(self.scale * torch.FloatTensor(target_distance).to(self.device) + self.bias, 0, 1)

            cond = -actor_scale * 230 + 230

            # target_flag = torch.FloatTensor(target_distance < self.mean_distance).to(self.device)

        # Get current Q estimates

        # mask calculation
        # cond = torch.where(target_distance - 0.25 > 0, 0.0 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        # cond = torch.where(target_distance - 0.15 < 0, 0.8 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        mask = (self.mask <= cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        actor_loss = (- self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)).mean()
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()



    def train_policy_quantile_with_uncertainty_v0(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            next_state_batch_np = next_state.cpu().numpy()
            next_action_batch_np = new_next_action.detach().cpu().numpy()
            query_data = np.concatenate([next_state_batch_np, next_action_batch_np], axis=1)
            tree_index = np.random.choice(len(kd_trees))
            kd_tree = kd_trees[tree_index]
            target_distance = kd_tree.query(query_data, k=1)[0]
            target_distance = torch.FloatTensor(target_distance).to(self.device)

            # target_flag = torch.FloatTensor(target_distance < self.mean_distance).to(self.device)

        # Get current Q estimates

        # mask calculation
        cond = torch.where(target_distance - 0.25 > 0, 0.0 * self.quantiles_total * self.base_tensor, 0.2 * self.quantiles_total * self.base_tensor)  # batch * 1
        cond = torch.where(target_distance - 0.15 < 0, 0.8 * self.quantiles_total * self.base_tensor, cond)  # batch * 1

        mask = (self.mask < cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy and alpha loss ---
        actor_loss = (- self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)).mean()
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()


    def train_policy_bc_with_uncertainty(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z = sorted_z[:, :self.quantiles_total - self.drop_quantile_bc]
            target = reward + not_done * self.discount * (sorted_z)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))
        combined_loss = source_loss + bc_loss

        state_batch_np = state.cpu().numpy()
        action_batch_np = self.actor(state).detach().cpu().numpy()
        query_data = np.concatenate([state_batch_np, action_batch_np], axis=1)
        #tree_index = np.random.choice(len(kd_trees))
        #kd_tree = kd_trees[tree_index]

        query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()

        target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)

        actor_scale = torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1)

        actor_loss = ((1-actor_scale) * source_loss).mean() + (actor_scale * bc_loss).mean()

        # --- Policy and alpha loss ---
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()


    def train_policy_bc_with_uncertainty_v1(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z = sorted_z[:, :self.quantiles_total - self.drop_quantile_bc]
            target = reward + not_done * self.discount * (sorted_z)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))
        combined_loss = source_loss + bc_loss

        #state_batch_np = state.cpu().numpy()
        #action_batch_np = self.actor(state).detach().cpu().numpy()
        #query_data = np.concatenate([state_batch_np, action_batch_np], axis=1)
        #tree_index = np.random.choice(len(kd_trees))
        #kd_tree = kd_trees[tree_index]

        # query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy() todo : last version
        query_data = self.feature_nn(state, self.actor(state)).detach().cpu().numpy()

        target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)

        # actor_scale = torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1)

        actor_scale = torch.clamp_(self.scale * torch.FloatTensor(target_distance).to(self.device) + self.bias, 0, 1)

        actor_loss = ((1-actor_scale) * source_loss).mean() + (actor_scale * bc_loss).mean()

        # --- Policy and alpha loss ---
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()


    def train_policy_bc_with_uncertainty_v0(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z = sorted_z[:, :self.quantiles_total - self.drop_quantile_bc]

            target = reward + not_done * self.discount * (sorted_z)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))
        combined_loss = source_loss + bc_loss

        state_batch_np = state.cpu().numpy()
        action_batch_np = self.actor(state).detach().cpu().numpy()
        query_data = np.concatenate([state_batch_np, action_batch_np], axis=1)
        tree_index = np.random.choice(len(kd_trees))
        kd_tree = kd_trees[tree_index]
        target_distance = kd_tree.query(query_data, k=1)[0]
        actor_scale = torch.FloatTensor((target_distance < 0.2).astype(float)).to(self.device) #  ((target_distance > 0.2).float() - 1) * 2

        actor_loss = (actor_scale * source_loss).mean() + ((1 - actor_scale) * combined_loss).mean()

        # --- Policy and alpha loss ---
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()

    def train_policy_bc_without_uncertainty(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z = sorted_z[:, :self.quantiles_total - self.drop_quantile_bc]
            target = reward + not_done * self.discount * (sorted_z)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))
        combined_loss = source_loss + bc_loss

        actor_loss = (combined_loss).mean()

        # --- Policy and alpha loss ---
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()

    def train_both(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()

            target_distance = kd_trees.query(query_data, k=1)[0] # / (self.state_dim + self.action_dim)
            # actor_scale = torch.clamp_(self.scale * torch.FloatTensor(target_distance).to(self.device) + self.bias, 0, 1)

            cond = -torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1) * 100 + 150

        mask = (self.mask <= cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))
        combined_loss = source_loss + bc_loss

        actor_loss = (combined_loss).mean()

        # --- Policy and alpha loss ---
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), actor_loss.item()



    def train_both_v1(self, memory, batch_size, kd_trees):
        state, action, next_state, reward, not_done = memory.sample(batch_size)

        with torch.no_grad():
            # get policy action
            new_next_action = self.actor_target(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            target = reward + not_done * self.discount * (sorted_z)

            query_data = self.feature_nn(next_state, new_next_action).detach().cpu().numpy()

            target_distance = kd_trees.query(query_data, k=1)[0] #  / (self.state_dim + self.action_dim)
            # actor_scale = torch.clamp_(self.scale * torch.FloatTensor(target_distance).to(self.device) + self.bias, 0, 1)

            cond = -torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1) * 100 + 150

        mask = (self.mask <= cond).float()  # batch * total_quantile

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, mask)

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = (self.actor(state) - action) ** 2
        original_actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True).mean()
        actor_loss = - self.critic(state, self.actor(state)).mean(2).mean(1, keepdim=True)
        source_loss = 2.5 * (actor_loss / (actor_loss.abs().mean().detach() + 1e-5))

        query_data = self.feature_nn(state, self.actor(state)).detach().cpu().numpy()

        target_distance = kd_trees.query(query_data, k=1)[0] / 20

        actor_scale = torch.clamp_(self.bc_scale * torch.FloatTensor(target_distance).to(self.device), 0, 1)

        actor_loss = ((1-actor_scale) * source_loss).mean() + (actor_scale * bc_loss).mean()


        # --- Policy and alpha loss ---
        if self.total_it % 2 == 0:

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        return critic_loss.item(), original_actor_loss.item()




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

