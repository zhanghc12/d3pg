import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from duelingpg.model import EnsembleDynamicsModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


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

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class WeightNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, ensemble_size):
        super(WeightNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, ensemble_size)

    def forward(self, state, action):
        w = F.relu(self.l1(torch.cat([state, action], 1)))
        w = F.relu(self.l2(w))
        return self.l3(w)

class EnsembleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, ensemble_size=5):
        super(EnsembleCritic, self).__init__()
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList()
        for i in range(ensemble_size):
            self.networks.append(nn.Sequential(nn.Linear(state_dim + action_dim, 256),
                                               nn.ReLU(),
                                               nn.Linear(256, 256),
                                               nn.ReLU(),
                                               nn.Linear(256, 1)))

    def forward(self, state, action):
        inputs = torch.cat([state, action], 1)
        outputs = []
        for network in self.networks:
            q = network(inputs)
            outputs.append(q)

        ensemble_q = torch.cat(outputs, dim=1)
        return ensemble_q


class DMPG(object):
    def __init__(self, state_dim, action_dim,
                 max_action, discount=0.99, tau=0.005, ensemble_size=5, env_name='Halfcheetah-v2'):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = EnsembleCritic(state_dim, action_dim, ensemble_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        num_networks = 7
        num_elites = 5

        self.model = EnsembleDynamicsModel(
            num_networks,
            num_elites,
            state_dim,
            action_dim,
            1,
            hidden_size=200,
            env_name=env_name
        )
        self.model.to(device)
        self.discount = discount
        self.tau = tau
        self.ensemble_size = ensemble_size

        self.weights_network = WeightNetwork(state_dim, action_dim, ensemble_size).to(device)
        self.weights_optimizer = torch.optim.Adam(self.weights_network.parameters(), lr=3e-4)
        '''
        self.weights = nn.Parameter(
            (torch.ones(
                (1,
                 self.ensemble_size)).float() /
                self.ensemble_size).to(device),
            requires_grad=True)
        '''

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(
            next_state, self.actor_target(next_state))
        target_weights = self.weights_network(next_state, self.actor_target(next_state))
        target_Q = torch.sum(target_Q * target_weights, dim=1, keepdim=True) / torch.sum(target_weights, dim=1, keepdim=True)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)
        current_Q = torch.mean(current_Q, dim=1, keepdim=True)


        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        # train model
        inputs, labels = replay_buffer.get_all_samples()
        '''
        self.model.train(
            inputs=inputs,
            labels=labels,
            batch_size=256,
            holdout_ratio=0.2,
            max_epochs_since_update=5)
        '''

        # after training, how to generate mc data to aid training
        n_step = 20
        self.n_step = n_step
        pred_state = state
        pred_reward_list = []
        pred_mc = torch.zeros_like(reward).to(device)
        terminal_flag = torch.zeros_like(reward, dtype=torch.bool).to(device)
        # done: terminal state adjustment, multiple state
        for i in range(self.n_step):
            pred_action = self.actor(pred_state)
            pred_next_state, pred_reward, pred_terminal, _ = self.model.step(
                pred_state.detach().numpy(), pred_action.detach().numpy())
            pred_next_state = torch.from_numpy(pred_next_state).to(device)
            pred_reward = torch.from_numpy(pred_reward).to(device)
            pred_terminal = torch.from_numpy(pred_terminal).to(device)

            pred_next_state = torch.where(terminal_flag, pred_state.float(), pred_next_state.float())
            pred_reward = torch.where(terminal_flag, 0., pred_reward.double())
            terminal_flag = terminal_flag | pred_terminal

            pred_state = pred_next_state
            pred_reward_list.append(pred_reward)
        for i in reversed(range(self.n_step)):
            pred_mc = self.discount * pred_mc + pred_reward_list[i]
        pred_mc = pred_mc.detach()
        pred_terminal = terminal_flag.detach()

        # todo: multiple trajectories to enable robustness
        pi_action = self.actor(state)
        ensemble_Q = self.critic(state, pi_action)
        aux_Q = torch.mean(ensemble_Q.detach(), dim=1, keepdim=True)

        pi_pred_action = self.actor(pred_state)
        target_ensemble_Q = self.critic_target(pred_state, pi_pred_action)
        weights = self.weights_network(pred_state, pi_pred_action)
        aux_target_Q = torch.sum(target_ensemble_Q.detach() * weights, dim=1, keepdim=True) / torch.sum(weights, dim=1, keepdim=True)

        weights_loss = torch.mean(
            torch.pow(aux_Q - pred_mc - (1 - pred_terminal.float()) * aux_target_Q, 2))

        self.weights_optimizer.zero_grad()
        weights_loss.backward()
        self.weights_optimizer.step()

        return actor_loss.item(), critic_loss.item(), weights_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
