import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from duelingpg import utils
from torch.distributions import Normal
# import torch.nn.utils.clip_gradnorm

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
        self.l4 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        mean = self.l3(q)
        sigma = self.l4(q)

        sigma = torch.log(1 + torch.exp(sigma)) + 1e-4
        sigma = torch.clamp_max(sigma, 5)
        # output_sig_pos = tf.log(1 + tf.exp(output_sig)) + 1e-06

        return mean, sigma

    def sample(self, state, action):
        mean, log_std = self.forward(state, action)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        log_prob = normal.log_prob(x_t)
        return x_t, log_prob, mean


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

        self.exp_actor = ExpActor(state_dim, action_dim, target_threshold).to(device)
        self.exp_actor_optimizer = torch.optim.Adam(self.exp_actor.parameters(), lr=3e-4)


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
        '''
        self.critics_eval = nn.ModuleList()
        for i in range(self.num_critic):
            self.critics_eval.append(Critic(state_dim, action_dim))
        self.critics_eval.to(device)

        self.critics_eval_target = copy.deepcopy(self.critics_eval)
        self.critic_eval_optimizer = torch.optim.Adam(self.critics_eval.parameters(), lr=3e-4)
        '''

        '''
        define the random reward 
        '''
        self.random_reward_net = LipRandomReward(state_dim, action_dim).to(device)
        self.random_reward_optimizer = torch.optim.Adam(self.random_reward_net.parameters(), lr=3e-4)

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
        if self.target_threshold > 0:
            return np.zeros_like(noise)

        exp_Qs = []
        exp_Qs_stds = []

        if self.version in [1,3]:
            for i in range(self.exp_num_critic):
                Q_mean, Q_log_std = self.exp_critics[i](state, action)
                exp_Qs.append(Q_mean)
                exp_Qs_stds.append(Q_log_std)
        elif self.version == 2:
            for i in range(self.num_critic):
                Q_mean, Q_log_std = self.critics[i](state, action)
                exp_Qs.append(Q_mean)
                exp_Qs_stds.append(Q_log_std)

        exp_Qs = torch.cat(exp_Qs, dim=1)
        exp_Qs_stds = torch.cat(exp_Qs_stds, dim=1)
        mean_exp_Qs = torch.mean(exp_Qs, dim=1)
        var_exp_Q = torch.sqrt((torch.mean(exp_Qs ** 2 + exp_Qs_stds, dim=1) - mean_exp_Qs ** 2).mean())

        action_grad = autograd.grad(var_exp_Q, action, retain_graph=True)[0]
        action_grad = action_grad / action_grad.norm()
        #print('here', action_grad, var_exp_Q, exp_Qs_stds, mean_exp_Qs, exp_Qs, action)

        #if torch.any(torch.isnan(action_grad)):
        #    print('here', action_grad, var_exp_Q, exp_Qs_stds, mean_exp_Qs, exp_Qs, action)
        return noise_scale * action_grad.cpu().data.numpy().flatten()

    def select_action(self, state, noisy=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if self.target_threshold > 0. and noisy == True:
            action = self.actor(state)
            return (action + self.exp_actor(state, action)).cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        constant_reward = torch.ones_like(reward).to(device)
        random_reward = self.random_reward_net(state, action)

        '''
        Compute critic loss
        '''
        # formulate target Q network
        target_Qs = []
        for i in range(self.num_critic):
            target_Q, _ = self.critics_target[i](next_state, self.actor_target(next_state))
            target_Qs.append(target_Q)
        target_Q = torch.min(torch.cat(target_Qs, dim=1), dim=1, keepdim=True)[0]
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # get the critic loss
        critic_loss = 0.
        for i in range(self.num_critic):
            current_Q_mean, current_Q_log_std = self.critics[i](state, action)
            # current_Q_inv_var = torch.exp(-current_Q_log_std)
            #critic_loss += torch.mean(torch.pow(current_Q_mean - target_Q, 2) * current_Q_inv_var)
            critic_loss += F.mse_loss(current_Q_mean, target_Q)
            # var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)

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
            exp_target_Q, _ = self.exp_critics_target[i](next_state, self.actor_target(next_state))
            exp_target_Qs.append(exp_target_Q)
        if self.version == 3:
            exp_target_Q = torch.max(torch.cat(exp_target_Qs, dim=1), dim=1, keepdim=True)[0]
        else:
            exp_target_Q = torch.mean(torch.cat(exp_target_Qs, dim=1), dim=1, keepdim=True)
        if self.exp_version == 0:
            exp_target_Q = reward + (not_done * self.discount * exp_target_Q).detach()
        elif self.exp_version == 1:
            exp_target_Q = constant_reward + (not_done * self.discount * exp_target_Q).detach()
        elif self.exp_version in [2, 3]:
            exp_target_Q = random_reward + (not_done * self.discount * exp_target_Q).detach()

        # get the critic loss
        exp_critic_loss = 0.
        exp_current_Q_log_stds = []
        for i in range(self.exp_num_critic):
            exp_current_Q_mean, exp_current_Q_log_std = self.exp_critics[i](state, action)
            exp_critic_loss += torch.mean(torch.pow(exp_current_Q_mean - exp_target_Q, 2) / exp_current_Q_log_std)
            exp_critic_loss += torch.mean(torch.log(exp_current_Q_log_std))
            exp_current_Q_log_stds.append(exp_current_Q_log_std)
        # print(exp_current_Q_log_stds)

        # Optimize the critic
        self.exp_critic_optimizer.zero_grad()
        exp_critic_loss.backward()
        # print('exp_critic_loss', exp_critic_loss)
        torch.nn.utils.clip_grad_norm_(self.exp_critics.parameters(), 1)
        self.exp_critic_optimizer.step()

        '''
        Compute actor loss
        '''
        pi_Qs = []
        for i in range(self.num_critic):
            pi_Q, _ = self.critics[i](state, self.actor(state))
            pi_Qs.append(pi_Q)
        pi_Qs = torch.mean(torch.cat(pi_Qs, dim=1), dim=1, keepdim=True)
        actor_loss = -(pi_Qs).mean()

        exp_actor_loss = torch.zeros([1])
        if self.target_threshold > 0:
            pi_action = self.actor(state).detach()
            exp_action = (pi_action + self.exp_actor(state, pi_action)).clamp_(-1, 1)

            exp_Qs = []
            exp_Qs_stds = []
            if self.version in [1, 3]:
                for i in range(self.exp_num_critic):
                    Q_mean, Q_log_std = self.exp_critics[i](state, exp_action)
                    exp_Qs.append(Q_mean)
                    exp_Qs_stds.append(Q_log_std)
            elif self.version == 2:
                for i in range(self.num_critic):
                    Q_mean, Q_log_std = self.critics[i](state, exp_action)
                    exp_Qs.append(Q_mean)
                    exp_Qs_stds.append(Q_log_std)
            else:
                raise NotImplementedError

            exp_Qs = torch.cat(exp_Qs, dim=1)
            exp_Qs_stds = torch.cat(exp_Qs_stds, dim=1)
            mean_exp_Qs = torch.mean(exp_Qs, dim=1)

            var_exp_Q = torch.sqrt((torch.mean(exp_Qs ** 2 + exp_Qs_stds, dim=1) - mean_exp_Qs ** 2).mean())
            # exp_actor_loss_2 = - std_Q.mean() / (std_Q.abs().mean().detach() + 1e-3)
            exp_actor_loss = -var_exp_Q

        if self.total_it % 2 == 0:

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.target_threshold > 0:
                # Optimize the actor
                self.exp_actor_optimizer.zero_grad()
                exp_actor_loss.backward()
                self.exp_actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critics.parameters(), self.critics_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.exp_critics.parameters(), self.exp_critics_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        '''
        compute random reward loss
        '''
        random_reward_loss = 0
        if self.exp_version == 2:
            state.requires_grad_(True)
            action.requires_grad_(True)

            random_reward = self.random_reward_net(state, action)
            # gradient clip
            random_reward_loss = utils.calc_gradient_penalty(state, random_reward) + utils.calc_gradient_penalty(action, random_reward)
            self.random_reward_optimizer.zero_grad()
            random_reward_loss.backward()
            self.random_reward_optimizer.step()
            random_reward_loss = random_reward_loss.item()

        return actor_loss.item(), critic_loss.item(), exp_critic_loss.item(), random_reward_loss, exp_actor_loss.item(), 0, 0, 0, 0


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