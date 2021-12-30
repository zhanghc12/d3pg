import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from duelingpg import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ExpActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ExpActor, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], dim=1)))
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


class LipRandomReward(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LipRandomReward, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        r = F.relu(self.l1(torch.cat([state, action], 1)))
        r = F.relu(self.l2(r))
        return self.l3(r)


class D3PG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, num_critic=2, exp_version=0, exp_num_critic=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        '''
        define the ensemble critics
        '''
        self.num_critic = num_critic
        self.critics = nn.ModuleList()
        for i in range(self.num_critic):
            self.critics.append(Critic(state_dim, action_dim))
        self.critics.to(device)

        self.critics_target = copy.deepcopy(self.critics)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.version = version
        self.target_threshold = target_threshold # note:
        self.total_it = 0

        self.exp_version = exp_version
        self.isnan = 0

    def select_noise(self, state, exp_noise):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        noise = np.random.normal(0, exp_noise, size=action.shape[1])
        noise_scale = np.linalg.norm(noise)
        if self.version == 0:
            return noise
        if self.target_threshold > 0:
            return np.zeros_like(noise)

        exp_Qs = []

        if self.version == 2:
            for i in range(self.num_critic):
                exp_Qs.append(self.critics[i](state, action))
        exp_Qs = torch.cat(exp_Qs, dim=1)
        std_Q = torch.std(exp_Qs, dim=1).mean()
        action_grad = autograd.grad(std_Q, action, retain_graph=True)[0]
        action_grad = action_grad / action_grad.norm()
        if torch.any(torch.isnan(action_grad)):
            self.isnan += 1
            return noise
        return noise_scale * action_grad.cpu().data.numpy().flatten()

    def select_action(self, state, noisy=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        '''
        Compute critic loss
        '''
        # formulate target Q network
        target_Qs = []
        for i in range(self.num_critic):
            target_Q = self.critics_target[i](next_state, self.actor_target(next_state))
            target_Qs.append(target_Q)
        target_Q = torch.min(torch.cat(target_Qs, dim=1), dim=1, keepdim=True)[0]
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # get the critic loss
        critic_loss = 0.
        for i in range(self.num_critic):
            current_Q = self.critics[i](state, action)
            critic_loss += F.mse_loss(current_Q, target_Q)
        adv_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        '''
        Compute actor loss
        '''
        pi_Qs = []
        for i in range(self.num_critic):
            pi_Q = self.critics[i](state, self.actor(state))
            pi_Qs.append(pi_Q)
        pi_Qs = torch.mean(torch.cat(pi_Qs, dim=1), dim=1, keepdim=True)
        actor_loss = -(pi_Qs).mean()

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


        return actor_loss.item(), critic_loss.item(), self.isnan, 0, 0, 0, 0, 0, 0


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