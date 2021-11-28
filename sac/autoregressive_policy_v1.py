import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

epsilon = 1e-6


def cosine_basis_functions(x, n_basis_functions=64):
    try:
        x = x.view(-1, 1)
    except:
        x = x.contiguous().view(-1, 1)

    i_pi = np.tile(np.arange(1, n_basis_functions + 1, dtype=np.float32), (x.shape[0], 1)) * np.pi
    i_pi = torch.Tensor(i_pi)
    if x.is_cuda:
        i_pi = i_pi.cuda()
    embedding = (x * i_pi).cos()
    return embedding


class CosineBasisLinear(nn.Module):
    def __init__(self, n_basis_functions, out_size):
        super(CosineBasisLinear, self).__init__()
        self.linear = nn.Linear(n_basis_functions, out_size)
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def forward(self, x):
        batch_size = x.shape[0]
        h = cosine_basis_functions(x, self.n_basis_functions)
        out = self.linear(h)
        out = out.view(batch_size, -1, self.out_size)
        return out


class AutoRegressiveStochasticActor(nn.Module):
    def __init__(self, num_inputs, action_dim, n_basis_functions=64):
        super(AutoRegressiveStochasticActor, self).__init__()
        self.action_dim = action_dim
        self.state_embedding = nn.Linear(num_inputs, 400)
        #cself.noise_embedding = CosineBasisLinear(n_basis_functions, 400)
        self.action_embedding = CosineBasisLinear(n_basis_functions, 400)

        self.rnn = nn.GRU(800, 400, batch_first=True)
        self.l1 = nn.Linear(400, 400)

        self.l_mean = nn.Linear(400, 1)
        self.l_logstd = nn.Linear(400, 1)

        #self._mean_W = nn.Parameter(torch.zeros(400, 1))
        #self._mean_b = nn.Parameter(torch.zeros(1,))

        #self._logstd_W = nn.Parameter(torch.zeros(400, 1))
        #self._logstd_b = nn.Parameter(torch.zeros(1, ))

        #self.l_mean = nn.Linear(400, action_dim)
        #self.l_logstd = nn.Linear(400, action_dim)

    def forward(self, state, actions=None):
        return self.supervised_forward(state, actions)

        if actions is not None:
            return self.supervised_forward(state, actions)
        batch_size = state.shape[0]
        # batch x 1 x 400
        state_embedding = F.leaky_relu(self.state_embedding(state)).unsqueeze(1)
        # batch x action dim x 400

        action_list = []
        action_means = []
        action_logstds = []

        action = torch.zeros(batch_size, 1)
        if state.is_cuda:
            action = action.cuda()
        hidden_state = None

        for idx in range(self.action_dim):
            # batch x 1 x 400
            action_embedding = F.leaky_relu(self.action_embedding(action.view(batch_size, 1, 1)))
            rnn_input = torch.cat([state_embedding, action_embedding], dim=2)
            gru_out, hidden_state = self.rnn(rnn_input, hidden_state)

            # batch x 6 * 400
            linear_out = F.leaky_relu(self.l1(gru_out.squeeze(1)))

            # batch * 6 * 1
            action_mean = self._mean_W[:, idx:idx+1] * linear_out + self._mean_b[idx:idx+1]
            action_mean = action_mean.squeeze()
            action_logstd = self._logstd_W[:, idx:idx+1] * linear_out + self._logstd_b[idx:idx+1]
            action_logstd = action_logstd.squeeze()

            action_means.append(action_mean)
            action_logstds.append(action_logstd)

            action_std = action_logstd.exp()
            normal = Normal(action_mean, action_std)
            action = normal.rsample()
            action = torch.tanh(action)

            action_list.append(action)

        actions = torch.stack(action_list, dim=1).squeeze(-1)
        means = torch.stack(action_means, dim=1).squeeze(-1)
        logstds = torch.stack(action_logstds, dim=1).squeeze(-1)

        return actions, means, logstds

    def supervised_forward(self, state, actions):
        # batch x action dim x 400
        state_embedding = F.leaky_relu(self.state_embedding(state)).unsqueeze(1).expand(-1, self.action_dim, -1)
        # batch x action dim x 400
        shifted_actions = torch.zeros_like(actions)
        shifted_actions[:, 1:] = actions[:, :-1]
        provided_action_embedding = F.leaky_relu(self.action_embedding(shifted_actions))

        rnn_input = torch.cat([state_embedding, provided_action_embedding], dim=2)
        gru_out, _ = self.rnn(rnn_input)
        linear_out = F.leaky_relu(self.l1(gru_out))

        # batch * 6 * 400
        # means = torch.matmul(linear_out, self._mean_W) + self._mean_b
        means = self.l_mean(linear_out)
        means = means.squeeze()
        # logstds = torch.matmul(linear_out, self._logstd_W) + self._logstd_b
        logstds = self.l_logstd(linear_out)

        logstds = logstds.squeeze()

        stds = logstds.exp()
        normal = Normal(means, stds)
        actions = normal.rsample()
        actions = torch.tanh(actions)

        return actions, means, logstds

    def log_prob(self, state, action):
        _, mean, log_std = self.supervised_forward(state, action)
        std = log_std.exp()
        normal = Normal(mean, std)
        # note: modified by zhc @0425

        u = 0.5 * torch.log((1 + epsilon + action) / (1 + epsilon - action))
        # before tanh

        log_prob = normal.log_prob(u)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return log_prob