import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

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
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0, target_threshold=0.1, num_critic=2, exp_version=0):
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

        '''
        define the exploration critics
        '''
        exp_num_critic = num_critic
        self.exp_num_critic = exp_num_critic
        self.exp_critics = nn.ModuleList()
        for i in range(self.exp_num_critic):
            self.exp_critics.append(Critic(state_dim, action_dim))
        self.exp_critics.to(device)

        self.exp_critics_target = copy.deepcopy(self.exp_critics)
        self.exp_critic_optimizer = torch.optim.Adam(self.exp_critics.parameters(), lr=3e-4)

        '''
        define the evaluation critics
        '''
        self.critics_eval = nn.ModuleList()
        for i in range(self.num_critic):
            self.critics_eval.append(Critic(state_dim, action_dim))
        self.critics_eval.to(device)

        self.critics_eval_target = copy.deepcopy(self.critics_eval)
        self.critic_eval_optimizer = torch.optim.Adam(self.critics_eval.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.version = version
        self.target_threshold = target_threshold # note:
        self.total_it = 0

        self.exp_version = exp_version

    def select_noise(self, state, exp_noise):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        noise = np.random.normal(0, exp_noise, size=action.shape[1])
        noise_scale = np.linalg.norm(noise)
        if self.version == 0:
            return noise

        exp_Qs = []
        if self.version == 1:
            for i in range(self.exp_num_critic):
                exp_Qs.append(self.exp_critics[i](state, action))
        elif self.version == 2:
            for i in range(self.num_critic):
                exp_Qs.append(self.critics[i](state, action))
        exp_Qs = torch.cat(exp_Qs, dim=1)
        std_Q = torch.std(exp_Qs, dim=1).mean()
        action_grad = autograd.grad(std_Q, action, retain_graph=True)[0]
        action_grad = action_grad / action_grad.norm()
        return noise_scale * action_grad.cpu().data.numpy().flatten()

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
        constant_reward = torch.ones_like(reward).to(device)

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
        optimize the exploration critic Q network, 
        exp version 0: original reward func
        exp version 1: constant reward func
        exp version 2: randomized reward func, not implemented yet
        '''
        exp_target_Qs = []
        for i in range(self.exp_num_critic):
            exp_target_Q = self.exp_critics_target[i](next_state, self.actor_target(next_state))
            exp_target_Qs.append(exp_target_Q)
        exp_target_Q = torch.mean(torch.cat(exp_target_Qs, dim=1), dim=1, keepdim=True)
        if self.exp_version == 0:
            exp_target_Q = reward + (not_done * self.discount * exp_target_Q).detach()
        else:
            exp_target_Q = constant_reward + (not_done * self.discount * exp_target_Q).detach()

        # get the critic loss
        exp_critic_loss = 0.
        for i in range(self.exp_num_critic):
            exp_current_Q = self.exp_critics[i](state, action)
            exp_critic_loss += F.mse_loss(exp_current_Q, exp_target_Q)

        # Optimize the critic
        self.exp_critic_optimizer.zero_grad()
        exp_critic_loss.backward()
        self.exp_critic_optimizer.step()

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

            for param, target_param in zip(self.exp_critics.parameters(), self.exp_critics_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return actor_loss.item(), critic_loss.item(), exp_critic_loss.item(), 0, 0, 0, 0, 0, 0


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