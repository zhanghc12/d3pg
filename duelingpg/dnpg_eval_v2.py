import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, num_critic=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.critic_eval = Critic(state_dim, action_dim).to(device)
        self.critic_eval_target = copy.deepcopy(self.critic_eval)
        self.critic_eval_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.version = version
        self.huber = torch.nn.SmoothL1Loss()

        self.target_threshold = target_threshold # note:
        self.total_it = 0
        self.use_terminal = True

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

    def eval_train_value(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size * 100)
        train_value, _ = self.critic(state, action)
        return train_value.mean().item()

    def train_first_phase(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        perturbed_next_state = next_state
        perturbed_reward = reward
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = self.actor_target(perturbed_next_state)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(perturbed_next_state, next_action)
            # target_Q = torch.min(target_Q1, target_Q2) # target_Q1 #
            # target_Q = (target_Q1 +  target_Q2) / 2
            target_Q = target_Q1
            target_Q = perturbed_reward + not_done * self.discount * target_Q
            # target_Q = target_Q + self.target_threshold * torch.abs(target_Q).mean() * torch.normal(mean=torch.zeros_like(target_Q), std=torch.ones_like(target_Q))

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean() # - self.critic.Q2(state, self.actor(state)).mean()

        # Delayed policy updates
        if self.total_it % 2 == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() - self.critic.Q2(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item(), 0, 0, 0, 0, 0


    def train_second_phase(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        perturbed_next_state = next_state + self.target_threshold * torch.normal(mean=torch.zeros_like(next_state), std=torch.ones_like(next_state))
        perturbed_reward = reward + self.target_threshold * torch.normal(mean=torch.zeros_like(reward), std=torch.ones_like(reward))

        # perturbed_next_state = next_state
        # perturbed_reward = reward
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = self.actor_target(perturbed_next_state)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(perturbed_next_state, next_action)
            # target_Q = torch.min(target_Q1, target_Q2) # target_Q1 #
            # target_Q = (target_Q1 +  target_Q2) / 2
            target_Q = target_Q1
            target_Q = perturbed_reward + not_done * self.discount * target_Q
            # target_Q = target_Q + self.target_threshold * torch.abs(target_Q).mean() * torch.normal(mean=torch.zeros_like(target_Q), std=torch.ones_like(target_Q))

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % 2 == 0:

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return 0, critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item(), 0, 0, 0, 0, 0

    def save_policy(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load_policy(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)