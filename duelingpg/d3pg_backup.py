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

        self.discount = discount
        self.tau = tau
        self.version = version
        self.huber = torch.nn.SmoothL1Loss()

        self.alpha_prime = torch.zeros(1, requires_grad=True).to(device)
        self.alpha_prime_optimizer = torch.optim.Adam([self.alpha_prime], lr=3e-4)

        self.log_beta_prime = torch.zeros(1, requires_grad=True).to(device)
        self.beta_prime_optimizer = torch.optim.Adam([self.log_beta_prime], lr=3e-4)

        self.target_threshold = target_threshold # note:
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        if self.version in [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15]:
            _, _, target_Q = self.critic_target(next_state, self.actor_target(next_state))
        elif self.version in [1, 5, 7]:
            target_Q, _, _ = self.critic_target(next_state, self.actor_target(next_state)) # assume the adv is
        else:
            raise NotImplementedError
        target_Q = reward + (not_done * self.discount * target_Q).detach()
        current_value, current_adv, current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        pi_value, pi_adv, pi_Q = self.critic(state, self.actor(state))

        if self.version == 11:
            pi_adv_with_fixed_feat = self.critic.forward_adv(state, self.actor(state))
            current_adv_with_fixed_feat = self.critic.forward_adv(state, action)
            pi_adv = pi_adv_with_fixed_feat

        if self.version in [4, 5]:
            beta_prime = self.log_beta_prime.exp()
            alpha_prime = torch.clamp(self.alpha_prime, min=-1000000.0, max=1000000.0)
            adv_loss = (alpha_prime * pi_adv).mean()
            prime_adv_loss = pi_adv.mean()
        elif self.version in [6, 7, 9, 11, 12]:
            alpha_prime = self.alpha_prime
            beta_prime = torch.clamp(self.log_beta_prime.exp(), min=0.0, max=1000000.0)
            adv_loss = beta_prime * torch.mean(torch.pow(pi_adv, 2))
            prime_adv_loss = torch.mean(torch.pow(pi_adv, 2))
        elif self.version == 3:
            alpha_prime = self.alpha_prime
            beta_prime = self.log_beta_prime.exp()
            adv_loss = self.huber(pi_adv, torch.zeros_like(pi_adv).to(device))
            prime_adv_loss = adv_loss
        else:
            alpha_prime = self.alpha_prime
            beta_prime = self.log_beta_prime.exp()

            adv_loss = torch.mean(torch.pow(pi_adv, 2))
            prime_adv_loss = adv_loss

        critic_loss = critic_loss + adv_loss

        if self.version == 14:
            target_v, target_adv, target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.discount * target_v).detach()
            current_value, current_adv, current_Q = self.critic(state, action)
            pi_value, pi_adv, pi_Q = self.critic(state, self.actor(state))
            current_Q = current_Q - pi_adv
            critic_loss = F.mse_loss(current_Q, target_Q)
            adv_loss = critic_loss

        if self.version == 15:
            target_v, target_adv, target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.discount * target_v).detach()
            current_value, current_adv, current_Q = self.critic(state, action)
            pi_value, pi_adv, pi_Q = self.critic(state, self.actor(state))
            # current_Q = current_Q - pi_adv
            critic_loss = F.mse_loss(current_Q, target_Q)
            adv_loss = critic_loss

        # plug into the consistency loss
        # advantage_diff = current_adv - pi_adv
        # Q_diff = current_Q - pi_Q = target_Q - pi_value
        if self.version == 12:
            beta_prime = torch.clamp(self.log_beta_prime.exp(), min=0.0, max=1000000.0)
            adv_diff = (current_adv - pi_adv).detach()
            consistency_loss = beta_prime * ((current_adv - adv_diff) ** 2).mean()
            critic_loss += consistency_loss
            # loss is zero, grad is zero, so we cannot do this thing better

        if self.version == 13:
            current_value, current_adv, current_Q = self.critic(state, action)
            pi_value, pi_adv, pi_Q = self.critic(state, self.actor(state))
            adv_diff = pi_adv - current_adv
            adv_diff = adv_diff.detach()
            consistency_loss = ((pi_adv - current_adv - adv_diff) ** 2).mean()
            consistency_loss.backward()
            # then we get the gradient on the correct track.
            # todo: keep track of the gradient of the critic
            l3_gradient = []

            alpha_prime = self.alpha_prime
            beta_prime = torch.clamp(self.log_beta_prime.exp(), min=0.0, max=1000000.0)
            adv_loss = beta_prime * torch.mean(torch.pow(pi_adv, 2))
            prime_adv_loss = torch.mean(torch.pow(pi_adv, 2))
            prime_adv_loss.backward()
            l2_gradient = []

            # l1_grad, l2_grad and l3_grad. l2 projected to the l3_grad and
            # l1_grad + (l3_grad * l2_grad) / l3
            # how to map the gradient

            # map the l2_gradient onto the l3_gradient surface


            # get the
            # calc the gradient of l2 on l3.


            # 12 is not correct, we must keep l2's gradient
            # l2 does not influence A(s,a) - A(s,\pi), make l2's gradient lie on the l3 surface.
            # l2 grad projects onto the l3 grad.


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.version in [4, 5]:
            _, pi_adv, _ = self.critic(state, self.actor(state))
            alpha_prime = torch.clamp(self.alpha_prime, min=-1000000.0, max=1000000.0)
            alpha_prime_loss = -(alpha_prime * pi_adv).mean()
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss.backward()
            self.alpha_prime_optimizer.step()

        if self.version in [6, 7, 9, 11, 12]:
            _, pi_adv, _ = self.critic(state, self.actor(state))
            beta_prime = torch.clamp(self.log_beta_prime.exp(), min=0.0, max=1000000.0)
            beta_prime_loss = -beta_prime * (torch.mean(torch.pow(pi_adv, 2)) - self.target_threshold).detach()
            self.beta_prime_optimizer.zero_grad()
            beta_prime_loss.backward()
            self.beta_prime_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state))[-1].mean()
        if self.version == 2:
            cur_value, cur_adv, cur_Q = self.critic(state, action)
            actor_loss += ((cur_adv > 0) * torch.pow(cur_adv.detach() - self.actor(state), 2)).mean()

        if self.version == 8 :
            cur_value, cur_adv, cur_Q = self.critic(state, action)
            actor_loss += ((cur_adv > 0) * torch.pow(action - self.actor(state), 2)).mean()

        if self.version == 9 :
            cur_value, cur_adv, cur_Q = self.critic(state, action)
            actor_loss += ((cur_adv > np.sqrt(self.target_threshold)) * torch.pow(action - self.actor(state), 2)).mean()

        if self.version == 10 :
            _, _, target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.discount * target_Q).detach()

            cur_value, cur_adv, cur_Q = self.critic(state, action)
            actor_loss += ((cur_Q > target_Q) * torch.pow(action - self.actor(state), 2)).mean()

        '''
        if self.version == 8:
            cur_value, cur_adv, cur_Q = self.critic(state, action)
            _, target_adv, target_Q = self.critic_target(next_state, self.actor_target(next_state))
            actor_loss += ((target_Q > 0) * torch.pow(action - self.actor(state), 2)).mean()
        '''

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), adv_loss.item(), prime_adv_loss.item(), alpha_prime.item(), beta_prime.item(), 0, 0, 0

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