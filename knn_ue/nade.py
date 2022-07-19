import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
class MaskedLinear(nn.Linear):
    """A Linear layer with masks that turn off some of the layer's weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones((out_features, in_features)))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class MADE():
    """The Masked Autoencoder Distribution Estimator (MADE) model."""
    def __init__(self, input_dim, embedding_dim, output_dim, hidden_dims=None, n_masks=1):
        """Initializes a new MADE instance.
        Args:
            input_dim: The dimensionality of the input. input_dim: state_dim output_dim: action_dim
            hidden_dims: A list containing the number of units for each hidden layer.
            n_masks: The total number of distinct masks to use during training/eval.
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._embedding_dim = embedding_dim

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, embedding_dim)

        self._dims = [self._embedding_dim] + (hidden_dims or []) + [self._output_dim]
        self._n_masks = n_masks
        self._mask_seed = 0

        layers = []
        layers.append(self.l1)
        layers.append(self.l2)
        layers.append(self.l3)

        for i in range(len(self._dims) - 1):
            in_dim, out_dim = self._dims[i], self._dims[i + 1]
            layers.append(MaskedLinear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers[-1] = nn.Sigmoid()  # Output is binary.
        self._net = nn.Sequential(*layers)

    def _sample_masks(self):
        """Samples a new set of autoregressive masks.
        Only 'self._n_masks' distinct sets of masks are sampled after which the mask
        sets are rotated through in the order in which they were sampled. In
        principle, it's possible to generate the masks once and cache them. However,
        this can lead to memory issues for large 'self._n_masks' or models many
        parameters. Finally, sampling the masks is not that computationally
        expensive.
        Returns:
            A tuple of (masks, ordering). Ordering refers to the ordering of the outputs
            since MADE is order agnostic.
        """
        rng = np.random.RandomState(seed=self._mask_seed % self._n_masks)
        self._mask_seed += 1

        # Sample connectivity patterns.
        conn = [rng.permutation(self._embedding_dim)]
        for i, dim in enumerate(self._dims[1:-1]):
            # NOTE(eugenhotaj): The dimensions in the paper are 1-indexed whereas
            # arrays in Python are 0-indexed. Implementation adjusted accordingly.
            low = 0 if i == 0 else np.min(conn[i - 1])
            high = self._embedding_dim - 1
            conn.append(rng.randint(low, high, size=dim))
        conn.append(np.copy(conn[0]))

        # Create masks.
        masks = [
            conn[i - 1][None, :] <= conn[i][:, None] for i in range(1, len(conn) - 1)
        ]
        masks.append(conn[-2][None, :] < conn[-1][:, None])

        return [torch.from_numpy(mask.astype(np.uint8)) for mask in masks], conn[-1]

    def _forward(self, x, masks):
        # If the input is an image, flatten it during the forward pass.
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(original_shape[0], -1)

        layers = [
            layer for layer in self._net.modules() if isinstance(layer, MaskedLinear)
        ]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)
        return self._net(x)# .view(original_shape)

    def forward(self, x):
        """Computes the forward pass.
        Args:
            x: Either a tensor of vectors with shape (n, input_dim) or images with shape
                (n, 1, h, w) where h * w = input_dim.
        Returns:
            The result of the forward pass.
        """

        masks, _ = self._sample_masks()
        return self._forward(x, masks)

    def sample(self, n_samples, conditioned_on=None):
        """See the base class."""
        if conditioned_on is None:
            shape = (n_samples, self._input_dim)
            conditioned_on = (torch.ones(shape) * -1).to(self.device)
        else:
            conditioned_on = conditioned_on.clone()

        with torch.no_grad():
            out_shape = conditioned_on.shape
            conditioned_on = conditioned_on.view(n_samples, -1)

            masks, ordering = self._sample_masks()
            ordering = np.argsort(ordering)
            for dim in ordering:
                out = self._forward(conditioned_on, masks)[:, dim]
                out = distributions.Bernoulli(probs=out).sample()
                conditioned_on[:, dim] = torch.where(
                    conditioned_on[:, dim] < 0, out, conditioned_on[:, dim]
                )
            return conditioned_on.view(out_shape)



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_bin=10):
        super(Critic, self).__init__()

        # state embedding
        self.input_dim = state_dim + action_dim + action_dim
        self.output_dim = num_bin
        self.l1 = nn.Linear(self.input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, self.output_dim)
        self.gap = 2. / num_bin

    def get_logprob(self, state, action):
        logits = []
        label = ((action + 1) // self.gap).long().detach()

        for i in range(action.shape[1]):
            action_mask = (torch.arange(action.shape[1]) < i).float().repeat(action.shape[0], 1).to(device)
            action_replaced = torch.where(action_mask > 0, action, torch.zeros_like(action)).to(device)
            one_hot = F.one_hot(torch.tensor([i]), num_classes=action.shape[1]).repeat(action.shape[0], 1).to(device)
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
        prob = torch.exp(logits)
        return prob

    def get_loss(self, state, action):
        loss = 0.
        label = ((action + 1) // self.gap).long().detach()

        for i in range(action.shape[1]):
            action_mask = (torch.arange(action.shape[1]) < i).float().repeat(action.shape[0], 1).to(device)
            action_replaced = torch.where(action_mask > 0, action, torch.zeros_like(action)).to(device)
            one_hot = F.one_hot(torch.tensor([i]), num_classes=action.shape[1]).repeat(action.shape[0], 1).to(device)
            input = torch.cat([state.float(), action_replaced.float(), one_hot.float()], dim=1)
            logit = F.relu(self.l1(input))
            logit = F.relu(self.l2(logit))
            logit = self.l3(logit)
            loss += nn.CrossEntropyLoss()(logit, label[:, i])
        return loss

    # how to calculate the density'

class AutoregressiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_bin=20):
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
        label = ((action + 1) // self.gap).long().detach()

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


    def predict(self, state):
        logits = []
        action = self.zero_action
        labels = []
        for i in range(2):
            action_replaced = torch.where(self.action_masks[i].float() > 0, action.float(), self.zero_action.float())
            one_hot = self.one_hot_list[i]
            input = torch.cat([state.float(), action_replaced, one_hot.float()], dim=1)
            logit = F.relu(self.l1(input))
            logit = F.relu(self.l2(logit))
            logit = self.l3(logit)
            logit = nn.LogSoftmax(dim=1)(logit)
            # logit = logit[label[:, i]]
            prediction = logit.argmax(dim=1)
            labels.append(prediction)
            action[:,i] = (prediction * self.gap - 1).detach()
        return labels

    def get_loss(self, state, action):
        loss = 0.
        label = ((action + 1) // self.gap).long().detach()

        for i in range(action.shape[1]):
            action_replaced = torch.where(self.action_masks[i].float() > 0, action.float(), self.zero_action.float())
            one_hot = self.one_hot_list[i]
            input = torch.cat([state.float(), action_replaced.float(), one_hot.float()], dim=1)
            logit = F.relu(self.l1(input))
            logit = F.relu(self.l2(logit))
            logit = self.l3(logit)
            loss += nn.CrossEntropyLoss()(logit, label[:, i])
        return loss


def test_autoregressive():
    num_bin = 100
    autoregressive_model = AutoregressiveModel(1, 2, num_bin=num_bin).to(device)
    #state = np.random.random(size=[256,10])
    #action = (np.random.random(size=[256,3]) - 0.5) * 2

    state = np.expand_dims(np.linspace(0, 1, 256), 1)
    action = np.concatenate([0.99 * np.sin(10 * state), -0.99 * np.cos(10 * state)], axis=1)

    print(state)

    state = torch.from_numpy(state).to(device)
    action = torch.from_numpy(action).to(device)
    print(autoregressive_model.get_logprob(state, action))
    #print(autoregressive_model.get_loss(state, action))
    optim = torch.optim.Adam(autoregressive_model.parameters(), lr=3e-4)

    for i in range(5000):
        loss = autoregressive_model.get_loss(state, action)
        loss = -torch.mean(autoregressive_model.get_logprob(state, action))
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(i, loss)
        print(torch.mean(autoregressive_model.get_logprob(state, action)))
        #print(torch.mean(autoregressive_model.get_logprob(state, action)))
    prediction = autoregressive_model.predict(state)
    print(autoregressive_model.predict(state))
    plt.plot(state.detach().numpy(), action.detach().numpy()[:,0])
    plt.plot(state.detach().numpy(), prediction[0].detach().numpy() * 2 / num_bin - 1)

    plt.plot(state.detach().numpy(), action.detach().numpy()[:,1])
    plt.plot(state.detach().numpy(), prediction[1].detach().numpy() * 2 / num_bin - 1)


    plt.show()




test_autoregressive()



