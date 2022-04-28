import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqc.spectral_normalization import spectral_norm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_sn=False):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        if use_sn:
            # Q1 architecture
            self.l1 = spectral_norm(nn.Linear(state_dim + action_dim, 256))
            self.l2 = spectral_norm(nn.Linear(256, 256))
            self.l3 = spectral_norm(nn.Linear(256, 1))

            # Q2 architecture
            self.l4 = spectral_norm(nn.Linear(state_dim + action_dim, 256))
            self.l5 = spectral_norm(nn.Linear(256, 256))
            self.l6 = spectral_norm(nn.Linear(256, 1))

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

class BiasCritic(nn.Module):
    def __init__(self, state_dim, use_sn=False):
        super(BiasCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class NoisyStateModel(nn.Module):
    def __init__(self, state_dim):
        super(NoisyStateModel, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, state_dim)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q1 = torch.tanh(q1)
        return q1

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


class D3PG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, num_critic=2, ratio=1):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, use_sn=(version==2)).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        #self.critic_eval = Critic(state_dim, action_dim).to(device)
        #self.critic_eval_target = copy.deepcopy(self.critic_eval)
        # self.critic_eval_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=3e-4)

        #self.bias_critic = BiasCritic(state_dim).to(device)
        #self.bias_critic_optimizer = torch.optim.Adam(self.bias_critic.parameters(), lr=3e-4)
        #self.bias_critic_loss = nn.MSELoss()

        self.nsm = NoisyStateModel(state_dim).to(device)
        self.nsm_optimizer = torch.optim.Adam(self.nsm.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.version = version
        self.huber = torch.nn.SmoothL1Loss()

        self.target_threshold = target_threshold # note:
        self.total_it = 0
        self.use_terminal = True
        self.ratio = ratio

        if self.version == 3:
            self.use_terminal = False


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def eval_value_clip(self, final_rewards, final_states, final_actions):
        # Sample replay buffer
        part = len(final_rewards) // 256
        part_train_values = []
        for i in range(part):
            part_states = torch.FloatTensor(final_states[i * 256: (i+1)*256, :]).to(device)
            part_actions = torch.FloatTensor(final_actions[i * 256: (i+1)*256, :]).to(device)
            part_train_value, _ = self.critic(part_states, part_actions)
            part_train_values.append(part_train_value.detach().cpu().numpy())

        eval_value = final_rewards[:part*256]
        train_value = np.squeeze(np.concatenate(part_train_values, axis=0))
        return np.mean(eval_value), np.mean(train_value), np.mean(train_value - eval_value), np.mean((train_value - eval_value) / (np.abs(eval_value) + 1e-3))


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        bias_loss = 0.
        bias_diff = 0.
        if self.version == 13:
            for i in range(10):
                state, action, next_state, reward, not_done, perturbed_next_state, perturbed_reward = replay_buffer.sample(
                    batch_size)

                with torch.no_grad():
                    perturbed_next_action = self.actor_target(perturbed_next_state)
                    perturbed_target_Q1, perturbed_target_Q2 = self.critic_target(perturbed_next_state, perturbed_next_action)
                    perturbed_target_Q = torch.min(perturbed_target_Q1, perturbed_target_Q2)

                    next_action = self.actor_target(next_state)
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)

                    label = target_Q - perturbed_target_Q

                prediction = self.bias_critic(perturbed_next_state)
                bias_critic_loss = self.bias_critic_loss(prediction, label)
                bias_diff = (label - prediction).mean().item()
                self.bias_critic_optimizer.zero_grad()
                bias_critic_loss.backward()
                self.bias_critic_optimizer.step()
                bias_loss = bias_critic_loss.item()

        if self.version == 17:
            for i in range(10):
                state, action, next_state, reward, not_done, perturbed_next_state, perturbed_reward = replay_buffer.sample(
                    batch_size)

                noise = self.nsm(perturbed_next_state)
                noisy_state = perturbed_next_state + 0.1 * self.target_threshold * noise
                noisy_action = self.actor_target(noisy_state)
                noisy_target_Q1_var, noisy_target_Q2_var = self.critic_target(noisy_state, noisy_action)
                noisy_Q = torch.min(noisy_target_Q1_var, noisy_target_Q2_var)
                nsm_loss = -noisy_Q.mean()
                self.nsm_optimizer.zero_grad()
                nsm_loss.backward()
                self.nsm_optimizer.step()

        # Sample replay buffer
        state, action, next_state, reward, not_done, perturbed_next_state, perturbed_reward = replay_buffer.sample(batch_size)


        if self.version == 12:
            with torch.no_grad():
                perturbed_next_action = self.actor_target(perturbed_next_state)
                perturbed_target_Q1, perturbed_target_Q2 = self.critic_target(perturbed_next_state, perturbed_next_action)
                perturbed_target_Q = torch.min(perturbed_target_Q1, perturbed_target_Q2)

                next_action = self.actor_target(next_state)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)

                label = target_Q - perturbed_target_Q

            prediction = self.bias_critic(perturbed_next_state)
            bias_critic_loss = self.bias_critic_loss(prediction, label)
            bias_diff = (label - prediction).mean().item()
            self.bias_critic_optimizer.zero_grad()
            bias_critic_loss.backward()
            self.bias_critic_optimizer.step()
            bias_loss = bias_critic_loss.item()

        if self.version == 15:
            noise = self.nsm(perturbed_next_state)
            #noisy_state = perturbed_next_state + self.target_threshold * noise
            # noisy_state = perturbed_next_state + 0.1 * self.target_threshold * noise
            #noisy_state = perturbed_next_state + 0.01 * self.target_threshold * noise
            #noisy_state = perturbed_next_state + 0.001 * self.target_threshold * noise
            noisy_state = perturbed_next_state + self.ratio * self.target_threshold * noise * replay_buffer.stds_gpu


            # self.ratio * self.target_threshold * noise * replay_buffer.stds_gpu
            noisy_action = self.actor_target(noisy_state)
            noisy_target_Q1_var, noisy_target_Q2_var = self.critic_target(noisy_state, noisy_action)
            noisy_Q = torch.min(noisy_target_Q1_var, noisy_target_Q2_var)
            nsm_loss = -noisy_Q.mean()
            self.nsm_optimizer.zero_grad()
            nsm_loss.backward()
            self.nsm_optimizer.step()


        if self.version == 22:
            noise = self.nsm(perturbed_next_state)
            #noisy_state = perturbed_next_state + self.target_threshold * noise
            # noisy_state = perturbed_next_state + 0.1 * self.target_threshold * noise
            #noisy_state = perturbed_next_state + 0.01 * self.target_threshold * noise
            #noisy_state = perturbed_next_state + 0.001 * self.target_threshold * noise
            noisy_state = perturbed_next_state + self.ratio * self.target_threshold * noise

            noisy_action = self.actor_target(noisy_state)
            noisy_target_Q1_var, noisy_target_Q2_var = self.critic_target(noisy_state, noisy_action)
            noisy_Q = torch.min(noisy_target_Q1_var, noisy_target_Q2_var)
            var_Q = torch.abs(noisy_target_Q1_var - noisy_target_Q2_var)
            nsm_loss = -noisy_Q.mean() + var_Q.mean()
            self.nsm_optimizer.zero_grad()
            nsm_loss.backward()
            self.nsm_optimizer.step()

        '''
        get target_Q
        '''
        if self.version in [5,6, 8]:
            next_state_var = Variable(perturbed_next_state, requires_grad=True)
            next_action_var = self.actor_target(next_state_var)
            target_Q1_var, target_Q2_var = self.critic_target(next_state_var, next_action_var)
            torch.min(target_Q1_var, target_Q2_var).sum().backward()
            next_state_grad = next_state_var.grad
            if self.version == 5:
                #approximate_state = perturbed_next_state + 0.1 * self.target_threshold * next_state_grad / (
                #            1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))
                approximate_state = perturbed_next_state + 0.033 * self.target_threshold * next_state_grad / (
                            1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))
            elif self.version == 6:
                #approximate_state = perturbed_next_state + 1 * self.target_threshold * next_state_grad / (
                #            1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))
                approximate_state = perturbed_next_state + 0.33 * self.target_threshold * next_state_grad / (
                            1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))

            elif self.version == 8:
                #approximate_state = perturbed_next_state + 10 * self.target_threshold * next_state_grad / (
                #            1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))
                approximate_state = perturbed_next_state + 0.01 * self.target_threshold * next_state_grad / (
                            1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))
            approximate_action = self.actor_target(approximate_state)
            approximate_target_Q1, approximate_target_Q2 = self.critic_target(approximate_state, approximate_action)
            target_Q = torch.min(approximate_target_Q1, approximate_target_Q2)
            target_Q = (perturbed_reward + not_done * self.discount * target_Q).detach()
            self.actor_target.zero_grad()
            self.critic_target.zero_grad()

        elif self.version in [14, 16]:

            approximate_state = perturbed_next_state

            for i in range(10):
                next_state_var = Variable(approximate_state, requires_grad=True)
                next_action_var = self.actor_target(next_state_var)
                target_Q1_var, target_Q2_var = self.critic_target(next_state_var, next_action_var)
                # (torch.min(target_Q1_var, target_Q2_var) - torch.mean((approximate_state - perturbed_next_state) ** 2, dim=1, keepdim=True)).sum().backward()
                (torch.min(target_Q1_var, target_Q2_var)).sum().backward()

                next_state_grad = next_state_var.grad
                if self.version == 14:
                    approximate_state = approximate_state + 0.1 * self.target_threshold * next_state_grad / (
                                1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))
                elif self.version == 16:
                    approximate_state = approximate_state + 0.01 * self.target_threshold * next_state_grad / (
                                1e-3 + torch.norm(next_state_grad, dim=1, keepdim=True))

                self.actor_target.zero_grad()
                self.critic_target.zero_grad()

            approximate_action = self.actor_target(approximate_state)
            approximate_target_Q1, approximate_target_Q2 = self.critic_target(approximate_state, approximate_action)
            target_Q = torch.min(approximate_target_Q1, approximate_target_Q2)
            target_Q = (perturbed_reward + not_done * self.discount * target_Q).detach()

        elif self.version in [15, 17]:
            #approximate_state = perturbed_next_state + 0.01 * self.target_threshold * self.nsm(perturbed_next_state)
            #approximate_state = perturbed_next_state + 0.001 * self.target_threshold * self.nsm(perturbed_next_state)
            #approximate_state = perturbed_next_state + 0.033 * self.target_threshold * self.nsm(perturbed_next_state)

            approximate_state = perturbed_next_state + self.ratio * self.target_threshold * self.nsm(perturbed_next_state) * replay_buffer.stds_gpu

            approximate_action = self.actor_target(approximate_state)
            approximate_target_Q1, approximate_target_Q2 = self.critic_target(approximate_state, approximate_action)
            target_Q = torch.min(approximate_target_Q1, approximate_target_Q2)
            target_Q = (perturbed_reward + not_done * self.discount * target_Q).detach()

        elif self.version == 22:
            #approximate_state = perturbed_next_state + 0.01 * self.target_threshold * self.nsm(perturbed_next_state)
            #approximate_state = perturbed_next_state + 0.001 * self.target_threshold * self.nsm(perturbed_next_state)
            # approximate_state = perturbed_next_state + 0.1 * self.target_threshold * self.nsm(perturbed_next_state)
            approximate_state = perturbed_next_state + self.ratio * self.target_threshold * self.nsm(perturbed_next_state)

            approximate_action = self.actor_target(approximate_state)
            approximate_target_Q1, approximate_target_Q2 = self.critic_target(approximate_state, approximate_action)
            target_Q = torch.min(approximate_target_Q1, approximate_target_Q2)
            target_Q = (perturbed_reward + not_done * self.discount * target_Q).detach()



        elif self.version in [18, 19]:
            next_action = self.actor_target(perturbed_next_state)
            target_Q1, target_Q2 = self.critic_target(perturbed_next_state, next_action)
            target_Q_mean = (target_Q1 + target_Q2) / 2
            target_Q_std = torch.abs(target_Q1 - target_Q2)
            ue = torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True))

            # how to scale ue to real ratio
            # when the ratio is 0.5, when the ratio is 0
            # when the distance is 0, the ratio is 0.5; when the distance is
            if self.version == 18:
                ratio = torch.clamp(-50 * ue + 0.5, 0, 0.5)
            if self.version == 19:
                ratio = torch.clamp(-5 * ue + 0.5, 0, 0.5)

            target_Q = target_Q_mean - ratio * target_Q_std
            target_Q = (perturbed_reward + not_done * self.discount * target_Q).detach()
        elif self.version in [20, 21]:
            next_action = self.actor_target(perturbed_next_state)
            target_Q1, target_Q2 = self.critic_target(perturbed_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q_mean = (target_Q1 + target_Q2) / 2
            target_Q_std = torch.abs(target_Q1 - target_Q2) + 1
            target_Q_std = target_Q_std / (torch.sum(target_Q_mean.abs()) + 1)

            cur_action = self.actor_target(state)
            cur_Q1, cur_Q2 = self.critic_target(state, cur_action)
            cur_Q_mean = (cur_Q1 + cur_Q2) / 2
            cur_Q_std = torch.abs(cur_Q1 - cur_Q2) + 1
            cur_Q_std = cur_Q_std / (torch.sum(cur_Q_mean.abs()) + 1)

            if self.version == 20:
                target_Q = target_Q + 0.1 * perturbed_reward * (1 / cur_Q_std - 1 / target_Q_std)
            if self.version == 21:
                target_Q = target_Q + 1 * perturbed_reward * (1 / cur_Q_std - 1 / target_Q_std)
                #target_Q = target_Q + 1 * perturbed_reward * (1 / cur_Q_std - 1 / target_Q_std)

            target_Q = (perturbed_reward + not_done * self.discount * target_Q).detach()

        else:
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                next_action = self.actor_target(perturbed_next_state)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(perturbed_next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)

                if self.version == 2:
                    target_Q = perturbed_reward / 1e3 + not_done * self.discount * target_Q
                elif self.version == 23:
                    target_Q = reward + not_done * self.discount * target_Q

                else:
                    target_Q = perturbed_reward + not_done * self.discount * target_Q

                if self.version == 3:
                    # target_Q = target_Q + 0.1 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)) * (target_Q.abs().mean())
                    target_Q = target_Q + 1 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)) #* (target_Q.abs().mean())

                if self.version == 4:
                    # target_Q = target_Q + 1 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)) * (target_Q.abs().mean())
                    target_Q = target_Q + 10 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)) #* (target_Q.abs().mean())

                if self.version in [9, 10, 11]:
                    if self.version == 9:
                        ratio = torch.clip(0.1 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)), 0, 0.5)
                    if self.version == 10:
                        ratio = torch.clip(1 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)), 0, 0.5)
                    if self.version == 11:
                        ratio = torch.clip(10 * torch.sqrt(torch.mean((perturbed_next_state - next_state) **2, dim=1, keepdim=True)), 0, 0.5)

                    target_Q = (1 - ratio) * torch.min(target_Q1, target_Q2) + ratio * torch.max(target_Q1, target_Q2)
                    target_Q = perturbed_reward + not_done * self.discount * target_Q

                if self.version in [12, 13]:
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = target_Q + self.bias_critic(perturbed_next_state).detach()
                    target_Q = perturbed_reward + not_done * self.discount * target_Q

        with torch.no_grad():
            test_noisy_next_action = self.actor(perturbed_next_state)
            test_noisy_target_Q1, test_noisy_target_Q2 = self.critic(perturbed_next_state, test_noisy_next_action)

            test_next_action = self.actor(next_state)
            test_target_Q1, test_target_Q2 = self.critic(next_state, test_next_action)

            q_diff = (test_target_Q1 - test_noisy_target_Q1).mean().item()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.version == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()\
                         -self.critic.Q1(perturbed_next_state, self.actor(perturbed_next_state)).mean() - self.critic.Q2(perturbed_next_state, self.actor(perturbed_next_state)).mean()
        else:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # Delayed policy updates
        if self.total_it % 2 == 0:

            # Compute actor losse
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item(), q_diff, bias_loss, bias_diff, 0, 0


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