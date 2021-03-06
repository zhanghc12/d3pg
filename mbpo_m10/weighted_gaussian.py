import torch


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import gzip
import itertools

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        data (np.ndarray): A numpy array containing the input
        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=True, use_disentangle=False, use_separate=False, use_kl=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)

        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.sn1 = EnsembleFC(state_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.sn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.sn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.sn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        # Add variance output
        self.sn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)


        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()
        self.use_disentangle = use_disentangle
        self.use_separate = use_separate
        self.use_kl = use_kl

        # self.nn6 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)


        # last layer is not a Gaussian, just two variables
        assert (self.use_disentangle and self.use_separate) is False

    def forward(self, x, y, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        sn1_output = self.swish(self.sn1(y))
        sn2_output = self.swish(self.sn2(sn1_output))
        sn3_output = self.swish(self.sn3(sn2_output))
        sn4_output = self.swish(self.sn4(sn3_output))
        sn5_output = self.sn5(sn4_output)

        sa_mean = nn5_output[:, :, :self.output_dim]
        s_mean = sn5_output[:, :, :self.output_dim]
        mean = sa_mean + s_mean

        sa_logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        sa_logvar = self.min_logvar + F.softplus(sa_logvar - self.min_logvar)

        s_logvar = self.max_logvar - F.softplus(self.max_logvar - sn5_output[:, :, self.output_dim:])
        s_logvar = self.min_logvar + F.softplus(s_logvar - self.min_logvar)

        logvar = torch.log(sa_logvar.exp() + s_logvar.exp())

        if ret_log_var:
            return mean, logvar, sa_mean, s_mean, sa_logvar, s_logvar
        else:
            return mean, torch.exp(logvar), sa_mean, s_mean, sa_logvar, s_logvar

    # def forward_s(self, x):
    #     sn1_output = self.swish(self.sn1(x))
    #     sn2_output = self.swish(self.sn2(sn1_output))
    #     sn3_output = self.swish(self.sn3(sn2_output))
    #     sn4_output = self.swish(self.sn4(sn3_output))
    #     sn5_output = self.sn5(sn4_output)
    #
    #     s_mean = sn5_output[:, :, :self.output_dim]
    #     s_logvar = self.max_logvar - F.softplus(self.max_logvar - sn5_output[:, :, self.output_dim:])
    #     s_logvar = self.min_logvar + F.softplus(s_logvar - self.min_logvar)
    #
    #     return s_mean, s_logvar, sn5_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    '''
    def get_dynamic_loss(self, iid_input, ood_input):
        dynamic_loss = 0.
        iid_mean, iid_logvar, iid_nn5_output = self.forward_s(iid_input)
        iid_var = torch.exp(iid_logvar)
        iid_prob = -((iid_mean - self.loc) ** 2) / (2 * iid_var) - iid_logvar - math.log(math.sqrt(2 * math.pi))
        iid_loss =
    '''

    def get_kl_loss(self, sa_mean, s_mean):
        # todo: take care of the dimension
        return torch.sum(torch.mean(torch.mean((sa_mean - s_mean.detach()) ** 2, dim=-1), dim=-1))

        # sigma_1 = logvar_1.exp().type(dtype=torch.float64)
        # sigma_2 = logvar_2.exp().type(dtype=torch.float64)
        # p_distribution = torch.distributions.MultivariateNormal(mu1, torch.diag_embed(sigma_1))
        # q_distribution = torch.distributions.MultivariateNormal(mu2, torch.diag_embed(sigma_2))
        # p_q_kl = torch.distributions.kl_divergence(p_distribution, q_distribution).mean()
        # loss = -p_q_kl
        # b = torch.tensor([1,1,1], dtype=torch.float64)
        # c = torch.tensor([2,2,2], dtype=torch.float64)
        #
        # b = b.type(dtype=torch.float64)
        # # b1 = torch.distributions.MultivariateNormal(b, torch.diag_embed(b))
        # # c1 = torch.distributions.MultivariateNormal(c, torch.diag_embed(c))
        # kl = torch.distributions.kl_divergence(b1, c1).mean()

    def get_disentangle_loss(self, sa_mean, s_mean):
        # sa_o = 0 then s_o = 0
        disentangle_loss = torch.mean(torch.mean(torch.abs(sa_mean * s_mean), dim=-1), dim=-1)
        return disentangle_loss

    def get_separate_loss(self, sa_mean, s_mean, sa_logvar, s_logvar, labels, inc_var_loss=True):
        s_inv_var = torch.exp(-s_logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            s_mse_loss = torch.mean(torch.mean(torch.pow(s_mean - labels, 2) * s_inv_var, dim=-1), dim=-1)
            s_var_loss = torch.mean(torch.mean(s_inv_var, dim=-1), dim=-1)
            s_total_loss = torch.sum(s_mse_loss) + torch.sum(s_var_loss)
        else:
            s_mse_loss = torch.mean(torch.pow(s_mean - labels, 2), dim=(1, 2))
            s_total_loss = torch.sum(s_mse_loss)

        labels = (labels - s_mean).detach()  # approximation error
        # then s, a things
        sa_inv_var = torch.exp(-sa_logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            sa_mse_loss = torch.mean(torch.mean(torch.pow(sa_mean - labels, 2) * sa_inv_var, dim=-1), dim=-1)
            sa_var_loss = torch.mean(torch.mean(sa_inv_var, dim=-1), dim=-1)
            sa_total_loss = torch.sum(sa_mse_loss) + torch.sum(sa_var_loss)
        else:
            sa_mse_loss = torch.mean(torch.pow(sa_mean - labels, 2), dim=(1, 2))
            sa_total_loss = torch.sum(sa_mse_loss)

        total_loss = s_total_loss + sa_total_loss
        mse_loss = s_mse_loss + sa_mse_loss
        return total_loss, mse_loss


    def loss(self, mean, logvar, labels, output, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3

        sa_mean, s_mean, sa_logvar, s_logvar = output
        if self.use_separate:
            total_loss, mse_loss = self.get_separate_loss(sa_mean, s_mean, sa_logvar, s_logvar, labels, inc_var_loss=inc_var_loss)
            return total_loss, mse_loss
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        if self.use_disentangle:
            total_loss += self.get_disentangle_loss(sa_mean, s_mean)
        if self.use_kl:
            total_loss += self.get_kl_loss(sa_mean, s_mean)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()


class WeightedGaussianEnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False, use_disentangle=False, use_separate=False, use_kl=False, writer=None):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay, use_disentangle=use_disentangle, use_separate=use_separate, use_kl=use_kl)
        self.scaler = StandardScaler()
        self.writer = writer
        self.step = 0

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar, *train_output = self.ensemble_model(train_input, train_input[:, :, :self.state_size], ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label, train_output)
                self.ensemble_model.train(loss)
                losses.append(loss)
                if epoch == 0 and start_pos == 0:
                    self.writer.add_scalar('loss/train_loss', loss.detach().cpu().numpy(), self.step)

            with torch.no_grad():
                holdout_mean, holdout_logvar, *holdout_output = self.ensemble_model(holdout_inputs, holdout_inputs[:, :, :self.state_size], ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, holdout_output, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
            if epoch == 0:
                self.writer.add_scalar('loss/holdout_loss', np.mean(holdout_mse_losses), self.step)
                self.writer.add_scalar('loss/top_holdout_loss', np.mean(holdout_mse_losses[self.elite_model_idxes]), self.step)

        self.step += 1

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var, *_ = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), input[None, :, :self.state_size].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
