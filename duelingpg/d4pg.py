import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from duelingpg.utils import MeanStdNormalizer

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

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2

class D4PG(object):
    def __init__(self, state_dim, action_dim, max_action, version, scale, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.total_it = 0
        self.ratio_mean_std = MeanStdNormalizer(device=device)
        self.scale = scale
        self.version = version

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        if self.version == 0:
            return self.train_td3(replay_buffer)
        if self.version == 1:
            return self.train_original(replay_buffer)
        elif self.version == 2:
            return self.train_no_target(replay_buffer)
        elif self.version == 3:
            return self.train_mixed_target(replay_buffer)


    def train_original(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2  = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = (((current_Q1 - target_Q) ** 2)).mean()

        '''
        the second copy to make sure no other stuff
        '''
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = critic_loss + (((current_Q2 - target_Q) ** 2)).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # the core issue is use the same data to update
        if self.total_it % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state,
                                                                                           self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), 0, 0, 0, 0, 0, 0, 0

    def train_td3(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2  = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = (((current_Q1 - target_Q) ** 2)).mean() + (((current_Q2 - target_Q) ** 2)).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # the core issue is use the same data to update
        if self.total_it % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state,
                                                                                           self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), 0, 0, 0, 0, 0, 0, 0



    def train_no_target_original(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2  = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        target_ratio = torch.sqrt(0.25 * (target_Q1 + target_Q2) ** 2 / ((target_Q1 - target_Q2) ** 2 + 1e-3)) # keep a
        target_ratio = self.ratio_mean_std(target_ratio.detach()) * self.scale
        # critic_loss = ((target_ratio + 1) * ((current_Q1 - target_Q) ** 2)).mean()

        critic_loss = (((current_Q1 - target_Q) ** 2)).mean()

        '''
        the second copy to make sure no other stuff
        '''
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        target_ratio = torch.sqrt(0.25 * (target_Q1 + target_Q2) ** 2 / ((target_Q1 - target_Q2) ** 2 + 1e-3))  # keep a
        target_ratio = self.ratio_mean_std(target_ratio.detach()) * self.scale
        # critic_loss = critic_loss + ((target_ratio + 1) * ((current_Q2 - target_Q) ** 2)).mean()
        critic_loss = critic_loss + (((current_Q2 - target_Q) ** 2)).mean()


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # the core issue is use the same data to update
        if self.total_it % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state,
                                                                                           self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), target_ratio.mean().item(), 0, 0, 0, 0, 0, 0


    def train_no_target(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2  = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = target_Q1
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = (((current_Q1 - target_Q) ** 2)).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        target_ratio = critic_loss
        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        # the core issue is use the same data to update
        if self.total_it % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), target_ratio.mean().item(), target_ratio.max().item(), target_ratio.min().item(), 0, 0, 0, 0

    def train_no_target_v2(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2  = self.critic(next_state, self.actor(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        target_ratio = torch.sqrt(0.25 * (target_Q1 + target_Q2) ** 2 / ((target_Q1 - target_Q2) ** 2 + 1e-3)) # keep a
        target_ratio = self.ratio_mean_std(target_ratio.detach()) * self.scale
        # critic_loss = ((target_ratio + 1) * ((current_Q1 - target_Q) ** 2)).mean()

        critic_loss = (((current_Q1 - target_Q) ** 2)).mean()

        '''
        the second copy to make sure no other stuff
        '''
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2 = self.critic(next_state, self.actor(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)

        target_ratio = torch.sqrt(0.25 * (target_Q1 + target_Q2) ** 2 / ((target_Q1 - target_Q2) ** 2 + 1e-3))  # keep a
        target_ratio = self.ratio_mean_std(target_ratio.detach()) * self.scale
        # critic_loss = critic_loss + ((target_ratio + 1) * ((current_Q2 - target_Q) ** 2)).mean()
        critic_loss = critic_loss + (((current_Q2 - target_Q) ** 2)).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # the core issue is use the same data to update
        if self.total_it % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state,
                                                                                           self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), target_ratio.mean().item(), target_ratio.max().item(), target_ratio.min().item(), 0, 0, 0, 0


    def train_mixed_target(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q1, target_Q2  = self.critic(next_state, self.actor(next_state))
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        target_ratio = torch.sqrt(0.25 * (target_Q1 + target_Q2) ** 2 / ((target_Q1 - target_Q2) ** 2 + 1e-3)) # keep a
        target_ratio = (self.ratio_mean_std(target_ratio.detach()) + 1 / 2) * self.scale
        critic_loss = (target_ratio * ((current_Q1 - target_Q) ** 2)).mean()

        target_Q1_original, target_Q2_original = self.critic_target(next_state, self.actor_target(next_state))
        target_Q_original = torch.min(target_Q1_original, target_Q2_original)
        target_Q_original = reward + (not_done * self.discount * target_Q_original).detach()
        critic_loss += ((1 - target_ratio) * ((current_Q1 - target_Q_original) ** 2)).mean()

        '''
        the second copy to make sure no other stuff
        '''
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) # note: we disard the
        # target_Q1, target_Q2 = self.critic(next_state, self.actor(next_state))
        # target_Q = torch.min(target_Q1, target_Q2)
        # target_Q = reward + (not_done * self.discount * target_Q).detach()
        #
        # current_Q1, current_Q2 = self.critic(state, action)
        # target_ratio = torch.sqrt(0.25 * (target_Q1 + target_Q2) ** 2 / ((target_Q1 - target_Q2) ** 2 + 1e-3))  # keep a
        # target_ratio = (self.ratio_mean_std(target_ratio.detach()) + 1 / 2) * self.scale
        critic_loss += (target_ratio * ((current_Q2 - target_Q) ** 2)).mean()

        # target_Q1_original, target_Q2_original = self.critic_target(next_state, self.actor_target(next_state))
        # target_Q_original = torch.min(target_Q1_original, target_Q2_original)
        # target_Q_original = reward + (not_done * self.discount * target_Q_original).detach()
        critic_loss += ((1 - target_ratio) * ((current_Q2 - target_Q_original) ** 2)).mean()


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # the core issue is use the same data to update
        if self.total_it % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state,
                                                                                           self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
