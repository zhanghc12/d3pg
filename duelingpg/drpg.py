import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from duelingpg.reverse_model import ReverseActor
from duelingpg.reverse_model import EnsembleDynamicsModel

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


class DRPG(object): # reverse model
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, env_name='', version=0):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau

        self.reverse_actor = ReverseActor(state_dim, action_dim, 1).to(device)
        self.reverse_actor_optimizer = torch.optim.Adam(self.reverse_actor.parameters(), lr=3e-4)

        num_networks = 7
        num_elites = 3
        self.reverse_model = EnsembleDynamicsModel(
            num_networks,
            num_elites,
            state_dim,
            action_dim,
            1,
            hidden_size=200,
            env_name=env_name,
            inner_epoch_num=10
        )
        self.reverse_model.to(device)

        self.cur_step = 0
        self.model_update_freq = 100
        self.version = version

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

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

        inputs, labels = replay_buffer.get_all_reverse_samples()
        if torch.cuda.is_available():
            if self.cur_step % self.model_update_freq == 0:
                self.reverse_model.train(
                    inputs=inputs,
                    labels=labels,
                    batch_size=256,
                    holdout_ratio=0.2,
                    max_epochs_since_update=5)

            log_prob = self.reverse_actor.log_prob(next_state, action)
            reverse_actor_loss = -log_prob.mean()
            self.reverse_actor_optimizer.zero_grad()
            reverse_actor_loss.backward()
            self.reverse_actor_optimizer.step()

        # the core idea is to find the Q(s,a) which and then resample and update acoording to td
        pred_states = replay_buffer.sample_states(batch_size=10 * batch_size)
        # evaluate and get pre post update, q value change is the biggest
        pred_actions = self.actor_target(pred_states)
        pre_Q = self.critic_target(pred_states, pred_actions)

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # todo: make sure not terminal
        post_Q = self.critic_target(pred_states, pred_actions)
        Q_diff = post_Q - pre_Q
        imp_index = torch.argsort(Q_diff, dim=0)[9 * batch_size:]
        # imp_states = torch.gather(pred_states, 0, imp_index) # pred_states[imp_index]
        imp_states = pred_states[imp_index.squeeze()]
        # largest Q difference, then we enumerate all the actions and states
        imp_actions = self.reverse_actor.sample(imp_states)[0]
        pre_imp_states, imp_rewards = self.reverse_model.step(imp_states.detach().cpu().numpy(), imp_actions.detach().cpu().numpy())
        imp_rewards = torch.from_numpy(imp_rewards).to(device).float()
        pre_imp_states = torch.from_numpy(pre_imp_states).to(device).float()

        # imp_target = torch.gather(post_Q, 0, imp_index) # post_Q[imp_index]
        imp_target = post_Q[imp_index.squeeze()]
        reverse_loss = F.mse_loss(self.critic(pre_imp_states, imp_actions), (imp_rewards + self.discount * imp_target).detach())
        self.critic_optimizer.zero_grad()
        reverse_loss.backward()
        self.critic_optimizer.step()

        self.cur_step += 1

        return actor_loss.item(), critic_loss.item(), 0, 0, 0, 0, 0, 0, 0

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
