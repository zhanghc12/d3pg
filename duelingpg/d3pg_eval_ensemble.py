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

    def forward(self, state, action):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value = self.lv(feat)
        adv = F.relu(self.l3(torch.cat([feat, action], 1)))
        adv = self.la(adv)

        return value, adv, value + adv


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
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, num_critic=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.num_critic = num_critic
        self.critics = nn.ModuleList()
        for i in range(self.num_critic):
            self.critics.append(DuelingCritic(state_dim, action_dim))
        self.critics.to(device)

        self.critics_target = copy.deepcopy(self.critics)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)

        self.critics_eval = nn.ModuleList()
        for i in range(self.num_critic):
            self.critics_eval.append(DuelingCritic(state_dim, action_dim))
        self.critics_eval.to(device)

        self.critics_eval_target = copy.deepcopy(self.critics_eval)
        self.critic_eval_optimizer = torch.optim.Adam(self.critics_eval.parameters(), lr=3e-4)

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

        target_vs = []
        for i in range(self.num_critic):
            target_v, _, _ = self.critics_eval_target[i](next_state, self.actor_target(next_state))
            target_vs.append(target_v)
        target_v = torch.mean(torch.cat(target_vs, dim=1), dim=1, keepdim=True)
        target_v = reward + (not_done * self.discount * target_v).detach()

        critic_loss = 0.
        for i in range(self.num_critic):
            pi_value, _, _ = self.critics_eval[i](state, self.actor(state))
            critic_loss += F.mse_loss(pi_value, target_v)

        # Optimize the critic
        self.critic_eval_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critics_eval.parameters(), self.critics_eval_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, critic_loss

    def train_value_mc(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done, timestep = replay_buffer.sample_include_timestep(batch_size)

        target_vs = []
        for i in range(self.num_critic):
            target_v, _, _ = self.critics_eval_target[i](next_state, self.actor_target(next_state))
            target_vs.append(target_v)
        target_v = torch.mean(torch.cat(target_vs, dim=1), dim=1, keepdim=True)
        target_v = reward + (not_done * torch.pow(self.discount, timestep) * target_v).detach()

        critic_loss = 0.
        for i in range(self.num_critic):
            pi_value, _, _ = self.critics_eval[i](state, self.actor(state))
            critic_loss += F.mse_loss(pi_value, target_v)

        # Optimize the critic
        self.critic_eval_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critics_eval.parameters(), self.critics_eval_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, critic_loss

    def eval_value(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size * 10)
        eval_values = []
        train_values = []
        for i in range(self.num_critic):
            eval_value, _, _ = self.critics_eval[i](state, action)
            train_value, _, _ = self.critics[i](state, action)
            eval_values.append(eval_value)
            train_values.append(train_value)

        eval_value = torch.mean(torch.cat(eval_values, dim=1), dim=1, keepdim=True)
        train_value = torch.mean(torch.cat(train_values, dim=1), dim=1, keepdim=True)

        return eval_value.mean().item(), train_value.mean().item(), (train_value - eval_value).mean().item()

    def train_original(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        target_vs = []
        #target_advs = []
        #target_Qs = []
        for i in range(self.num_critic):
            target_v, target_adv, target_Q = self.critics_target[i](next_state, self.actor_target(next_state))
            target_vs.append(target_v)
            #target_advs.append(target_adv)
            #target_Qs.append(target_Q)

        target_v = torch.mean(torch.cat(target_vs, dim=1), dim=1, keepdim=True)
        target_Q = reward + (not_done * self.discount * target_v).detach()

        critic_loss = 0.
        for i in range(self.num_critic):
            current_value, current_adv, current_Q = self.critics[i](state, action)
            pi_value, pi_adv, pi_Q = self.critics[i](state, self.actor(state))
            current_Q = current_Q - pi_adv
            critic_loss += F.mse_loss(current_Q, target_Q)
        adv_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        pi_advs = []
        for i in range(self.num_critic):
            pi_value, pi_adv, pi_Q = self.critics[i](state, self.actor(state))
            pi_advs.append(pi_Q)
        pi_advs = torch.min(torch.cat(pi_advs, dim=1), dim=1, keepdim=True)[0]
        # pi_advs = torch.mean(torch.cat(pi_advs, dim=1), dim=1, keepdim=True)
        actor_loss = -(pi_advs).mean()

        if self.total_it % 2 == 0:

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), adv_loss.item(), 0, 0, 0, 0, 0, 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        target_vs = []
        #target_advs = []
        #target_Qs = []
        for i in range(self.num_critic):
            target_v, target_adv, target_Q = self.critics_target[i](next_state, self.actor_target(next_state))
            target_vs.append(target_Q)
            #target_advs.append(target_adv)
            #target_Qs.append(target_Q)

        target_v = torch.mean(torch.cat(target_vs, dim=1), dim=1, keepdim=True)
        target_Q = reward + (not_done * self.discount * target_v).detach()

        critic_loss = 0.
        for i in range(self.num_critic):
            current_value, current_adv, current_Q = self.critics[i](state, action)
            pi_value, pi_adv, pi_Q = self.critics[i](state, self.actor(state))
            current_Q = current_Q - pi_adv
            critic_loss += F.mse_loss(current_Q, target_Q)
        adv_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        pi_advs = []
        for i in range(self.num_critic):
            pi_value, pi_adv, pi_Q = self.critics[i](state, self.actor(state))
            pi_advs.append(pi_Q)
        pi_advs = torch.min(torch.cat(pi_advs, dim=1), dim=1, keepdim=True)[0]
        # pi_advs = torch.mean(torch.cat(pi_advs, dim=1), dim=1, keepdim=True)
        actor_loss = -(pi_advs).mean()

        if self.total_it % 1 == 0:

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
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