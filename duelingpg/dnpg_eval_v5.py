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
    def __init__(self, state_dim, action_dim, use_sn=False):
        super(BiasCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action, next_state):
        inputs = torch.cat([state, action, next_state], dim=1)
        q1 = F.relu(self.l1(inputs))
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

        self.bias_critic = BiasCritic(state_dim, action_dim).to(device)
        self.bias_critic_optimizer = torch.optim.Adam(self.bias_critic.parameters(), lr=3e-4)
        self.bias_critic_loss = nn.MSELoss()

        #self.nsm = NoisyStateModel(state_dim).to(device)
        #self.nsm_optimizer = torch.optim.Adam(self.nsm.parameters(), lr=3e-4)

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

        if self.version in [110, 111]:
            iters = 1
            if self.version == 111:
                iters = 10
            for i in range(iters):
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

                prediction = self.bias_critic(state, action, perturbed_next_state)
                bias_critic_loss = self.bias_critic_loss(prediction, label)
                bias_diff = (label - prediction).mean().item()
                self.bias_critic_optimizer.zero_grad()
                bias_critic_loss.backward()
                self.bias_critic_optimizer.step()
                bias_loss = bias_critic_loss.item()

        # Sample replay buffer
        state, action, next_state, reward, not_done, perturbed_next_state, perturbed_reward = replay_buffer.sample(batch_size)


        # train
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            if self.version in [104, 106, 109]:
                next_action = self.actor_target(next_state)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                # target_Q = (target_Q1 + target_Q2) / 2

                target_Q = reward + not_done * self.discount * target_Q
            elif self.version == 108:
                target_Q  = reward

            else:
                next_action = self.actor_target(perturbed_next_state)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(perturbed_next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                # target_Q = (target_Q1 + target_Q2) / 2

                target_Q = perturbed_reward + not_done * self.discount * target_Q

                if self.version in [110, 111]:
                    target_Q += not_done * self.discount * self.bias_critic(state, action, perturbed_next_state).detach()


        with torch.no_grad():
            test_noisy_next_action = self.actor(perturbed_next_state)
            test_noisy_target_Q1, test_noisy_target_Q2 = self.critic(perturbed_next_state, test_noisy_next_action)

            test_next_action = self.actor(next_state)
            test_target_Q1, test_target_Q2 = self.critic(next_state, test_next_action)

            q_diff = (test_target_Q1 - test_noisy_target_Q1).mean().item()

            '''
            perturbed_next_state_v1 = next_state + self.target_threshold * next_state * ((torch.normal(torch.zeros_like(next_state), torch.ones_like(next_state)) > 0).float() - 0.5 ) * 2
            test_noisy_next_action_v1 = self.actor(perturbed_next_state_v1)
            test_noisy_target_Q1_v1, test_noisy_target_Q2_v1 = self.critic(perturbed_next_state_v1, test_noisy_next_action_v1)
            q_diff_v1 = (test_target_Q1 - test_noisy_target_Q1_v1).mean().item() # no estimate

            perturbed_next_state_v2 = (1 + self.target_threshold) * next_state - self.target_threshold * next_state * ((torch.normal(torch.zeros_like(next_state), torch.ones_like(next_state)) > 0).float() - 0.5 ) * 2
            test_noisy_next_action_v2 = self.actor(perturbed_next_state_v2)
            test_noisy_target_Q1_v2, test_noisy_target_Q2_v2 = self.critic(perturbed_next_state_v2, test_noisy_next_action_v2)
            q_diff_v2 = (test_target_Q1 - test_noisy_target_Q1_v2).mean().item() # over estimate

            perturbed_next_state_v3 = next_state + self.target_threshold * torch.normal(torch.zeros_like(next_state), torch.abs(next_state)) # 107
            test_noisy_next_action_v3 = self.actor(perturbed_next_state_v3)
            test_noisy_target_Q1_v3, _ = self.critic(perturbed_next_state_v3, test_noisy_next_action_v3)
            q_diff_v3 = (test_target_Q1 - test_noisy_target_Q1_v3).mean().item() # no estimate

            perturbed_next_state_v4 = (1 + self.target_threshold) * next_state - self.target_threshold * torch.normal(torch.zeros_like(next_state), torch.abs(next_state)) # 107
            test_noisy_next_action_v4 = self.actor(perturbed_next_state_v4)
            test_noisy_target_Q1_v4, _ = self.critic(perturbed_next_state_v4, test_noisy_next_action_v4)
            q_diff_v4 = (test_target_Q1 - test_noisy_target_Q1_v4).mean().item() # over estimate

            perturbed_next_state_v5 = next_state + self.target_threshold * torch.normal(torch.zeros_like(next_state),
                                                                                        torch.ones_like(next_state))  # 107
            test_noisy_next_action_v5 = self.actor(perturbed_next_state_v5)
            test_noisy_target_Q1_v5, _ = self.critic(perturbed_next_state_v5, test_noisy_next_action_v5)
            q_diff_v5 = (test_target_Q1 - test_noisy_target_Q1_v5).mean().item() # under estimate

            perturbed_next_state_v6 = (1 + self.target_threshold) * next_state - self.target_threshold * torch.normal(
                torch.zeros_like(next_state), torch.ones_like(next_state))  # 107
            test_noisy_next_action_v6 = self.actor(perturbed_next_state_v6)
            test_noisy_target_Q1_v6, _ = self.critic(perturbed_next_state_v6, test_noisy_next_action_v6)
            q_diff_v6 = (test_target_Q1 - test_noisy_target_Q1_v6).mean().item()


            perturbed_next_state_v7 = next_state + self.target_threshold * next_state
            test_noisy_next_action_v7 = self.actor(perturbed_next_state_v7)
            test_noisy_target_Q1_v7, _ = self.critic(perturbed_next_state_v7, test_noisy_next_action_v7)
            q_diff_v7 = (test_target_Q1 - test_noisy_target_Q1_v7).mean().item() # under estimate

            perturbed_next_state_v8 = next_state - self.target_threshold * next_state
            test_noisy_next_action_v8 = self.actor(perturbed_next_state_v8)
            test_noisy_target_Q1_v8, _ = self.critic(perturbed_next_state_v8, test_noisy_next_action_v8)
            q_diff_v8 = (test_target_Q1 - test_noisy_target_Q1_v8).mean().item() # under estimate

            perturbed_next_state_v9 = (1 - self.target_threshold) * next_state + self.target_threshold * next_state * ((torch.normal(torch.zeros_like(next_state), torch.ones_like(next_state)) > 0).float() - 0.5 ) * 2
            test_noisy_next_action_v9 = self.actor(perturbed_next_state_v9)
            test_noisy_target_Q1_v9, _ = self.critic(perturbed_next_state_v9, test_noisy_next_action_v9)
            q_diff_v9 = (test_target_Q1 - test_noisy_target_Q1_v9).mean().item() # over estimate

            perturbed_next_state_v10 = (1 - self.target_threshold) * next_state + self.target_threshold * torch.normal(torch.zeros_like(next_state), torch.abs(next_state)) # 107
            test_noisy_next_action_v10 = self.actor(perturbed_next_state_v10)
            test_noisy_target_Q1_v10, _ = self.critic(perturbed_next_state_v10, test_noisy_next_action_v10)
            q_diff_v10 = (test_target_Q1 - test_noisy_target_Q1_v10).mean().item() # over estimate
            '''

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

        # Delayed policy updates
        if self.total_it % 2 == 0:

            # Compute actor losse
            # Optimize the actor
            if self.version in [103, 105, 106, 108]:
                pass
            else:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item(), q_diff, bias_loss, bias_diff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
               # q_diff_v1, q_diff_v2, q_diff_v3, q_diff_v4, q_diff_v5, q_diff_v6, q_diff_v7, q_diff_v8, q_diff_v9, q_diff_v10



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