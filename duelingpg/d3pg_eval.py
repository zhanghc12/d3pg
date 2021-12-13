import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingCritic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.lv = nn.Linear(256, 1)

        self.l3 = nn.Linear(256 + action_dim, 256)
        self.la = nn.Linear(256, 1)


        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.lv2 = nn.Linear(256, 1)

        self.l6 = nn.Linear(256 + action_dim, 256)
        self.la2 = nn.Linear(256, 1)

    def forward(self, state, action):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value = self.lv(feat)
        adv = F.relu(self.l3(torch.cat([feat, action], 1)))
        adv = self.la(adv)

        feat2 = F.relu(self.l5(F.relu(self.l4(state))))
        value2 = self.lv2(feat2)
        adv2 = F.relu(self.l6(torch.cat([feat2, action], 1)))
        adv2 = self.la2(adv2)

        return value, adv, value + adv, value2, adv2, value2 + adv2


class DuelingCriticv2(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DuelingCriticv2, self).__init__()

        self.l1 = nn.Linear(num_inputs, 256)
        self.l2 = nn.Linear(256, 256)
        self.lv_1 = nn.Linear(256, 1)

        self.l3 = nn.Linear(num_inputs + num_actions, 256)
        self.l4 = nn.Linear(256, 256)
        self.la_1 = nn.Linear(256, 1)


    def forward(self, state, action, return_full=False):
        sa = torch.cat([state, action], 1)
        value_1 = self.lv_1(F.relu(self.l2(F.relu(self.l1(state)))))
        adv_1 = self.la_1(F.relu(self.l4(F.relu(self.l3(sa)))))

        q1 = adv_1 + value_1
        return value_1, adv_1, q1

    def get_value(self, state):
        value_1 = self.lv_1(F.relu(self.l2(F.relu(self.l1(state)))))
        return value_1


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


class D3PG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = DuelingCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.critic_eval = DuelingCritic(state_dim, action_dim).to(device)
        self.critic_eval_target = copy.deepcopy(self.critic_eval)
        self.critic_eval_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=3e-4)


        self.discount = discount
        self.tau = tau
        self.version = version
        self.huber = torch.nn.SmoothL1Loss()

        self.target_threshold = target_threshold # note:
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train_value(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        target_v1, _, _, target_v2, _, _ = self.critic_eval_target(next_state, self.actor_target(next_state))
        target_v = reward + (not_done * self.discount * (target_v1 + target_v2) / 2).detach()
        pi_value1, _, _, pi_value2, _, _ = self.critic_eval(state, action)
        critic_loss = F.mse_loss(pi_value1, target_v) + F.mse_loss(pi_value2, target_v)

        # Optimize the critic
        self.critic_eval_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic_eval.parameters(), self.critic_eval_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, ((pi_value1 + pi_value2) / 2).mean().item()

    def train_value_mc(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done, timestep = replay_buffer.sample_include_timestep(batch_size)

        target_v1, _, _, target_v2, _, _ = self.critic_eval_target(next_state, self.actor_target(next_state))
        target_v = reward + (not_done * torch.pow(self.discount, timestep) * (target_v1 + target_v2) / 2).detach()
        pi_value1, _, _, pi_value2, _, _ = self.critic_eval(state, action)
        critic_loss = F.mse_loss(pi_value1, target_v) + F.mse_loss(pi_value2, target_v)


        # Optimize the critic
        self.critic_eval_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic_eval.parameters(), self.critic_eval_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, ((pi_value1 + pi_value2) / 2).mean().item()

    def eval_value(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size * 10)
        eval_value1, _, _, eval_value2, _, _ = self.critic_eval(state, action)
        train_value1, _, _, train_value2, _, _ = self.critic(state, action)

        eval_value = (eval_value1 + eval_value2)/2
        train_value = (train_value1 + train_value2)/2

        return eval_value.mean().item(), train_value.mean().item(), (train_value - eval_value).mean().item()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        target_v1, target_adv1, target_Q1, target_v2, target_adv2, target_Q2, = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * (target_v1 + target_v2) / 2).detach()

        current_value1, current_adv1, current_Q1, current_value2, current_adv2, current_Q2 = self.critic(state, action)
        pi_value1, pi_adv1, pi_Q1,  pi_value2, pi_adv2, pi_Q2 = self.critic(state, self.actor(state))
        current_Q1 = current_Q1 - pi_adv1
        current_Q2 = current_Q2 - pi_adv2
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        adv_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss

        pi_value1, pi_adv1, pi_Q1,  pi_value2, pi_adv2, pi_Q2 = self.critic(state, self.actor(state))
        # actor_loss = -torch.min(pi_adv1, pi_adv2).mean()
        actor_loss = ((pi_adv1 + pi_adv2) / 2).mean()

        if self.total_it % 1 == 0:

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), adv_loss.item(), 0, 0, 0, 0, 0, 0

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