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

    def forward_adv(self, state, action):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        feat = feat.detach()
        adv = F.relu(self.l3(torch.cat([feat, action], 1)))
        adv = self.la(adv)
        return adv

    def get_value(self, state):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value = self.lv(feat)
        return value


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

        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        self.qf_criterion = nn.MSELoss()
        self.num_critic = 5
        for i in range(self.num_critic):
            self.critics.append(DuelingCritic(
                state_dim=state_dim,
                action_dim=action_dim))

        self.critic_targets = copy.deepcopy(self.critics)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)

        self.critics = self.critics.to(device)
        self.critic_targets = self.critic_targets.to(device)
        self.discount = discount
        self.tau = tau
        self.version = version
        self.huber = torch.nn.SmoothL1Loss()

        '''
        self.alpha_prime = torch.zeros(1, requires_grad=True).to(device)
        self.alpha_prime_optimizer = torch.optim.Adam([self.alpha_prime], lr=3e-4)

        self.log_beta_prime = torch.zeros(1, requires_grad=True).to(device)
        self.beta_prime_optimizer = torch.optim.Adam([self.log_beta_prime], lr=3e-4)
        '''
        self.target_threshold = target_threshold # note:
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        next_obs = next_state
        rewards = reward
        obs = state
        actions = action

        target_vs = []

        for i in range(self.num_critic):
            target_vs.append(self.critic_targets[i].get_value(next_obs))

        target_q_values = torch.mean(torch.squeeze(torch.cat(target_vs, dim=-1)), dim=-1, keepdim=True)  # [0]
        q_target = rewards + not_done * self.discount * target_q_values
        q_target = q_target.detach()

        qf_loss = 0.
        alpha_prime_loss = 0.
        for i in range(self.num_critic):
            v_pred, adv_pred, q_pred = self.critics[i](obs, actions)
            _, adv_pi, _ = self.critics[i](obs, self.actor(obs))
            # todo: define mask, the i*bs: (i+1)*bs is 1, other is zeros
            # masks = torch.zeros_like(rewards.to(ptu.device))
            # masks[i * 256: (i+1) * 256] = 1
            # qf_loss += (masks * (q_pred - adv_pi - q_target) ** 2).mean()
            qf_loss += self.qf_criterion(q_pred - adv_pi, q_target)

        critic_loss = qf_loss
        adv_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor_loss = -self.critic(state, self.actor(state))[-1].mean()

        advs = []
        pi_action = self.actor(state)
        for i in range(self.num_critic):
            advs.append(self.critics[i](state, pi_action)[-1])  # -2 to -1

        advs = torch.squeeze(torch.cat(advs, dim=-1))
        actor_loss = -torch.mean(advs, dim=-1)  # [0]
        actor_loss = actor_loss.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        pi_value, pi_adv, pi_Q = self.critics[0](state, self.actor(state))
        current_value, current_adv, current_Q = self.critics[0](state, action)
        target_v, target_adv, target_Q = self.critic_targets[0](next_state, self.actor_target(next_state))

        # Update the frozen target models
        for param, target_param in zip(self.critics.parameters(), self.critic_targets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), adv_loss.item(), pi_Q.mean().item(), pi_Q.max().item(), \
               pi_Q.min().item(), current_Q.mean().item(), current_Q.max().item(), current_Q.min().item(), \
               current_adv.mean().item(), current_adv.max().item(), current_adv.min().item(), \
               current_value.mean().item(), current_value.max().item(), current_value.min().item(), \
               target_v.mean().item(), target_v.max().item(), target_v.min().item(),


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