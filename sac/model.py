import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class DuelingNetworkv2(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DuelingNetworkv2, self).__init__()

        self.l1 = nn.Linear(num_inputs, 256)
        self.l2 = nn.Linear(256, 256)
        self.lv_1 = nn.Linear(256, 1)

        self.l3 = nn.Linear(num_inputs + num_actions, 256)
        self.l4 = nn.Linear(256, 256)
        self.la_1 = nn.Linear(256, 1)

        self.l5 = nn.Linear(num_inputs, 256)
        self.l6 = nn.Linear(256, 256)
        self.lv_2 = nn.Linear(256, 1)

        self.l7 = nn.Linear(num_inputs + num_actions, 256)
        self.l8 = nn.Linear(256, 256)
        self.la_2 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action, return_full=False):
        sa = torch.cat([state, action], 1)
        value_1 = self.lv_1(F.relu(self.l2(F.relu(self.l1(state)))))
        adv_1 = self.la_1(F.relu(self.l4(F.relu(self.l3(sa)))))

        q1 = adv_1 + value_1

        value_2 = self.lv_2(F.relu(self.l6(F.relu(self.l5(state)))))
        adv_2 = self.la_2(F.relu(self.l8(F.relu(self.l7(sa)))))

        q2 = adv_2 + value_2

        if return_full:
            return value_1, adv_1, q1, value_2, adv_2, q2
        return q1, q2

    def get_value(self, state):
        value_1 = self.lv_1(F.relu(self.l2(F.relu(self.l1(state)))))
        value_2 = self.lv_2(F.relu(self.l6(F.relu(self.l5(state)))))
        return value_1, value_2

    def get_adv(self, state, action):
        sa = torch.cat([state, action], 1)
        adv_1 = self.la_1(F.relu(self.l4(F.relu(self.l3(sa)))))
        adv_2 = self.la_2(F.relu(self.l8(F.relu(self.l7(sa)))))
        return adv_1, adv_2



class DuelingNetworkv0(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DuelingNetworkv0, self).__init__()

        self.l1 = nn.Linear(num_inputs, 256)
        self.l2 = nn.Linear(256, 256)
        self.lv_1 = nn.Linear(256, 1)

        self.l3 = nn.Linear(256 + num_actions, 256)
        self.l4 = nn.Linear(256, 256)
        self.la_1 = nn.Linear(256, 1)

        self.l5 = nn.Linear(num_inputs, 256)
        self.l6 = nn.Linear(256, 256)
        self.lv_2 = nn.Linear(256, 1)

        self.l7 = nn.Linear(256 + num_actions, 256)
        self.l8 = nn.Linear(256, 256)
        self.la_2 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action, return_full=False):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value_1 = self.lv_1(feat)
        adv_1 = F.relu(self.l4(F.relu(self.l3(torch.cat([feat, action], 1)))))
        adv_1 = self.la_1(adv_1)
        q1 = adv_1 + value_1

        feat = F.relu(self.l6(F.relu(self.l5(state))))
        value_2 = self.lv_2(feat)
        adv_2 = F.relu(self.l8(F.relu(self.l7(torch.cat([feat, action], 1)))))
        adv_2 = self.la_2(adv_2)
        q2 = adv_2 + value_2

        if return_full:
            return value_1, adv_1, q1, value_2, adv_2, q2
        return q1, q2

    def get_value(self, state):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value_1 = self.lv_1(feat)

        feat = F.relu(self.l6(F.relu(self.l5(state))))
        value_2 = self.lv_2(feat)

        return value_1, value_2

    def get_adv(self, state, action):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        adv_1 = F.relu(self.l4(F.relu(self.l3(torch.cat([feat, action], 1)))))
        adv_1 = self.la_1(adv_1)

        feat = F.relu(self.l6(F.relu(self.l5(state))))
        adv_2 = F.relu(self.l8(F.relu(self.l7(torch.cat([feat, action], 1)))))
        adv_2 = self.la_2(adv_2)

        return adv_1, adv_2

class DuelingNetworkv1(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DuelingNetworkv1, self).__init__()

        self.l1 = nn.Linear(num_inputs, 256)
        self.l2 = nn.Linear(256, 256)
        self.lv_1 = nn.Linear(256, 1)

        self.l3 = nn.Linear(256 + num_actions, 256)
        self.l4 = nn.Linear(256, 256)
        self.la_1 = nn.Linear(256, 1)

        self.l5 = nn.Linear(num_inputs, 256)
        self.l6 = nn.Linear(256, 256)
        self.lv_2 = nn.Linear(256, 1)

        self.l7 = nn.Linear(256 + num_actions, 256)
        self.l8 = nn.Linear(256, 256)
        self.la_2 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action, return_full=False):
        feat = F.relu(self.l1(state))
        value_1 = self.lv_1(F.relu(self.l2(feat)))
        adv_1 = F.relu(self.l4(F.relu(self.l3(torch.cat([feat, action], 1)))))
        adv_1 = self.la_1(adv_1)
        q1 = adv_1 + value_1

        feat = F.relu(self.l5(state))
        value_2 = self.lv_2(F.relu(self.l6(feat)))
        adv_2 = F.relu(self.l8(F.relu(self.l7(torch.cat([feat, action], 1)))))
        adv_2 = self.la_2(adv_2)
        q2 = adv_2 + value_2

        if return_full:
            return value_1, adv_1, q1, value_2, adv_2, q2
        return q1, q2

    def get_value(self, state):
        feat = F.relu(self.l1(state))
        value_1 = self.lv_1(F.relu(self.l2(feat)))

        feat = F.relu(self.l5(state))
        value_2 = self.lv_2(F.relu(self.l6(feat)))

        return value_1, value_2

    def get_adv(self, state, action):
        feat = F.relu(self.l1(state))
        adv_1 = F.relu(self.l4(F.relu(self.l3(torch.cat([feat, action], 1)))))
        adv_1 = self.la_1(adv_1)

        feat = F.relu(self.l5(state))
        adv_2 = F.relu(self.l8(F.relu(self.l7(torch.cat([feat, action], 1)))))
        adv_2 = self.la_2(adv_2)

        return adv_1, adv_2



class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = action
        x_t = atanh(y_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return log_prob

    def get_entropy(self, state):
        mean, log_std = self.forward(state)
        entropy = (np.log(2 * np.pi) + 1) * mean.shape[1] / 2 + log_std.sum(dim=1, keepdim=True)
        return entropy


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)



# class AutoregressivePolicy(nn.Module):
