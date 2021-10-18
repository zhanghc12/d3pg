import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class Hyper_QNetwork(nn.Module):
    # Hyper net create weights with respect to the state and estimates function Q_s(a)
    def __init__(self, meta_v_dim, base_v_dim):
        super(Hyper_QNetwork, self).__init__()

        dynamic_layer = 256
        z_dim = 1024

        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # Q function net
        self.layer1 = Head(z_dim, base_v_dim, dynamic_layer, sttdev=0.05)
        self.last_layer = Head(z_dim, dynamic_layer, 1, sttdev=0.008)

    def forward(self, meta_v, base_v, debug=None):
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1, b1, s1 = self.layer1(z)
        w2, b2, s2 = self.last_layer(z)

        if debug is not None:
            debug['w1'][-1].append(w1.detach().clone().cpu().numpy())
            debug['w2'][-1].append(w2.detach().clone().cpu().numpy())
            debug['b1'][-1].append(b1.detach().clone().cpu().numpy())
            debug['b2'][-1].append(b2.detach().clone().cpu().numpy())
            debug['s1'][-1].append(s1.detach().clone().cpu().numpy())
            debug['s2'][-1].append(s2.detach().clone().cpu().numpy())

        # dynamic network pass
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = torch.bmm(w2, out) * s2 + b2

        return out.view(-1, 1)

class Hyper_Critic(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self, state_dim, action_dim, num_hidden=1):
        super(Hyper_Critic, self).__init__()
        meta_v_dim = state_dim
        base_v_dim = action_dim
        self.q1 = Hyper_QNetwork(meta_v_dim, base_v_dim)
        self.q2 = Hyper_QNetwork(meta_v_dim, base_v_dim)

    def forward(self, state, action, debug=None):
        q1 = self.q1(state, action, debug)
        q2 = self.q2(state, action)
        return q1,q2

    def Q1(self, state, action, debug):
        q1 = self.q1(state, action, debug)
        return q1


class ResBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class Head(nn.Module):
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super(Head, self).__init__()

        h_layer = 1024
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out

        self.W1 = nn.Linear(h_layer, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(h_layer, output_dim_out)
        self.s1 = nn.Linear(h_layer, output_dim_out)

        self.init_layers(sttdev)

    def forward(self, x):
        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)
        s = 1. + self.s1(x).view(-1, self.output_dim_out, 1)

        return w, b, s

    def init_layers(self, stddev):
        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.s1.bias)
        torch.nn.init.zeros_(self.b1.bias)


class Meta_Embadding(nn.Module):

    def __init__(self, meta_dim, z_dim):
        super(Meta_Embadding, self).__init__()
        self.z_dim = z_dim
        self.hyper = nn.Sequential(

            nn.Linear(meta_dim, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),

            nn.Linear(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),

            nn.Linear(512, 1024),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),

        )

        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -bound, bound)

class HyperNetwork(nn.Module): # s vs s to form a feature
    def __init__(self, meta_v_dim, base_v_dim, output_dim):
        super(HyperNetwork, self).__init__()

        dynamic_layer = 256
        z_dim = 1024

        self.output_dim = output_dim

        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # Q function net
        self.layer1 = Head(z_dim, base_v_dim, dynamic_layer, sttdev=0.05)
        self.last_layer = Head(z_dim, dynamic_layer, output_dim, sttdev=0.008)

    def forward(self, meta_v, base_v, debug=None):
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1, b1, s1 = self.layer1(z)
        w2, b2, s2 = self.last_layer(z)

        # dynamic network pass
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = torch.bmm(w2, out) * s2 + b2

        return out.view(-1, self.output_dim)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.hyper = HyperNetwork(state_dim, state_dim, action_dim)
        #self.l1 = nn.Linear(256, 256)
        #self.l2 = nn.Linear(256, 256)
        #self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = self.hyper(state, state)
        #a = F.relu(self.hyper(state, state))
        #a = F.relu(self.l1(a))
        #a = F.relu(self.l2(a))
        #return self.max_action * torch.tanh(self.l3(a))
        return self.max_action * torch.tanh(a)

class OriginalActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(OriginalActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))



class OriginalCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(OriginalCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.hyper = HyperNetwork(state_dim + action_dim, state_dim + action_dim, 1)
        #self.l1 = nn.Linear(256, 256)
        #self.l2 = nn.Linear(256, 256)
        #self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        inputs = torch.cat([state, action], 1)
        q = self.hyper(inputs, inputs)
        #q = F.relu(self.hyper(inputs, inputs))
        #q = F.relu(self.l1(q))
        #q = F.relu(self.l2(q))
        #return self.l3(q)
        return q


class DHPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, version=0):
        if version == 0:
            self.actor = OriginalActor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)
        elif version == 1:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)
        elif version == 2:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.critic = OriginalCritic(state_dim, action_dim).to(device)
        else:
            self.actor = OriginalActor(state_dim, action_dim, max_action).to(device)
            self.critic = OriginalCritic(state_dim, action_dim).to(device)

        #self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        #self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.version = version

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), 0

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
