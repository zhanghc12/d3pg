import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sf.utils import soft_update, hard_update
from sf.model import GaussianPolicy
from sf.utils import *
from tqc.spectral_normalization import spectral_norm


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

        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = self.l3(q2)
        return q2


class SharedSF(nn.Module):
    def __init__(self, state_dim, action_dim, feat_dim, hidden_dim):
        super(SharedSF, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim

        self.feat_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)  # todo: feature is naive
        self.hidden_l1 = nn.Linear(hidden_dim, hidden_dim)

        self.psi_l1 = nn.Linear(hidden_dim, self.feat_dim)
        self.phi_l1 = nn.Linear(hidden_dim, self.feat_dim)
        self.r_l1 = nn.Linear(self.feat_dim, 1)

        init_weights_xavier(self)

    def forward(self, state, action):
        return self.get_qvalue(state, action)

    def get_feature(self, state, action):
        input = torch.cat([state, action], dim=1)
        feat = F.relu(self.feat_l1(input))
        feat = F.relu(self.hidden_l1(feat))
        return feat

    def local_embedding(self, state, action):
        feat = self.get_feature(state, action)
        phi = self.phi_l1(feat)
        norm = phi.norm(dim=-1, keepdim=True) + 1e-6
        return (phi / norm).view(-1, self.feat_dim)

    def global_embedding(self, state, action):
        feat = self.get_feature(state, action)
        psi = self.psi_l1(feat)
        return psi

    def get_reward(self, state, action):
        phi = self.local_embedding(state, action)
        reward = self.r_l1(phi)
        return reward

    def get_w(self):
        return self.r_l1.weight, self.r_l1.bias,

    def get_qvalue(self, state, action):
        psi = self.global_embedding(state, action)
        return self.r_l1(psi)

class IdpSF(nn.Module):
    def __init__(self, state_dim, action_dim, feat_dim, hidden_dim):
        super(IdpSF, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim

        # first layer feature
        self.weight_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.weight_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.weight_l3 = nn.Linear(hidden_dim, self.feat_dim) # w : 1 * feat_dim

        # self.weight = nn.Parameter(torch.zeros(self.feat_dim, 1), requires_grad=True)

        self.weight = nn.Linear(self.feat_dim, 1) # w : 1 * feat_dim

        self.phi_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.phi_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.phi_l3 = nn.Linear(hidden_dim, self.feat_dim) # phi: 1 * feat_dim

        self.psi_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.psi_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.psi_l3 = nn.Linear(hidden_dim, self.feat_dim) # psi: 1 * feat_dim

    def forward(self, state, action, fix_feature=True):
        # get successor feature of (state, action) pair: w(s,a): reward
        input = torch.cat([state, action], dim=1)
        #w = F.relu(self.weight_l1(input))
        #w = F.relu(self.weight_l2(w))
        #w = self.weight_l3(w)

        psi = F.relu(self.psi_l1(input))
        psi = F.relu(self.psi_l2(psi))
        psi = self.psi_l3(psi)

        # Q = (w * psi).sum(dim=1, keepdim=True)
        Q = self.weight(psi)
        return Q

    def forward_reward(self, state, action, fix_feature=True):
        # get successor feature of (state, action) pair: w(s,a): reward
        # input = torch.cat([state, action], dim=1)
        #w = F.relu(self.weight_l1(input))
        #w = F.relu(self.weight_l2(w))
        #w = self.weight_l3(w)

        phi = self.get_phi(state, action)
        if fix_feature:
            phi = phi.detach()

        # Q = (w * phi).sum(dim=1, keepdim=True)
        Q = self.weight(phi)
        return Q

    def get_phi(self, state, action):
        input = torch.cat([state, action], dim=1)
        phi = F.relu(self.phi_l1(input))
        phi = F.relu(self.phi_l2(phi))
        phi = F.relu(self.phi_l3(phi))
        phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-6)
        return phi

    def get_psi(self, state, action):
        input = torch.cat([state, action], dim=1)
        psi = F.relu(self.psi_l1(input))
        psi = F.relu(self.psi_l2(psi))
        psi = self.psi_l3(psi)
        return psi

    def get_w(self, state, action):
        input = torch.cat([state, action], dim=1)
        w = F.relu(self.weight_l1(input))
        w = F.relu(self.weight_l2(w))
        w = self.weight_l3(w)
        return w


class MixedSF(nn.Module):
    def __init__(self, state_dim, action_dim, feat_dim, hidden_dim):
        super(MixedSF, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim

        # first layer feature + hidden layer
        self.feature_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.feature_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.feature_l3 = nn.Linear(hidden_dim, hidden_dim) # w : 1 * feat_dim

        # psi layer
        self.psi_l1 = nn.Linear(hidden_dim, self.feat_dim)
        # self.psi_l2 = nn.Linear(hidden_dim, hidden_dim)
        # self.psi_l3 = nn.Linear(hidden_dim, self.feat_dim) # psi: 1 * feat_dim

        # reward layer
        self.phi_l1 = nn.Linear(hidden_dim, self.feat_dim)
        self.weight_l1 = nn.Linear(self.feat_dim, 1)

    def get_feature(self, state, action):
        input = torch.cat([state, action], dim=1)
        feature = F.relu(self.feature_l1(input))
        feature = F.relu(self.feature_l2(feature))
        feature = F.relu(self.feature_l3(feature))
        return feature

    def forward(self, state, action, fix_feature=True):
        feature = self.get_feature(state, action)
        psi = F.relu(self.psi_l1(feature))
        Q = self.weight_l1(psi)
        return Q

    def get_reward(self, state, action):
        feature = self.get_feature(state, action)
        phi = F.relu(self.phi_l1(feature))
        phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-6)
        R = self.weight_l1(phi)
        return R

    def get_phi(self, state, action):
        feature = self.get_feature(state, action)
        phi = F.relu(self.phi_l1(feature))
        phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-6)
        return phi

    def get_psi(self, state, action):
        feature = self.get_feature(state, action)
        psi = F.relu(self.psi_l1(feature))
        return psi


# first, reimplement the bonus
class SepSF(nn.Module):
    def __init__(self, state_dim, action_dim, feat_dim, hidden_dim):
        super(SepSF, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim

        # phi layer
        #self.feature_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        #self.feature_l2 = nn.Linear(hidden_dim, hidden_dim)
        #self.feature_l3 = nn.Linear(hidden_dim, self.feat_dim) # w : 1 * feat_dim
        sn = True
        if sn:
            # return spectral_norm(F.relu(self.l3(q)), norm_bound=0.95, n_power_iterations=1)  # todo: if relu or not

            self.feature_l1 = spectral_norm(nn.Linear(self.state_dim + self.action_dim, hidden_dim), norm_bound=0.95, n_power_iterations=1)
            self.feature_l2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim), norm_bound=0.95, n_power_iterations=1)
            self.feature_l3 = spectral_norm(nn.Linear(hidden_dim, self.feat_dim), norm_bound=0.95, n_power_iterations=1)   # w : 1 * feat_dim


        else:
            self.feature_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
            self.feature_l2 = nn.Linear(hidden_dim, hidden_dim)
            self.feature_l3 = nn.Linear(hidden_dim, self.feat_dim)  # w : 1 * feat_dim

        # psi layer
        self.psi_l1 = nn.Linear(self.feat_dim, hidden_dim)
        self.psi_l2 = nn.Linear(hidden_dim, self.feat_dim)

        # weight layer
        self.weight_l1 = nn.Linear(self.feat_dim, 1)

        # reward layer
        self.reward_l1 = nn.Linear(self.feat_dim, 1)

        # state layer todo : nomralization
        self.forward_l1 = nn.Linear(self.feat_dim, hidden_dim)
        self.forward_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.forward_l3 = nn.Linear(hidden_dim, hidden_dim)
        self.forward_l4 = nn.Linear(hidden_dim, state_dim + 1)

    def get_phi(self, state, action):
        input = torch.cat([state, action], dim=1)
        phi = F.relu(self.feature_l1(input))
        phi = F.relu(self.feature_l2(phi))
        phi = F.relu(self.feature_l3(phi))
        # phi = spectral_norm(phi, norm_bound=0.95, n_power_iterations=1)
        phi = phi / (phi.norm(dim=-1, keepdim=True, p=1) + 1e-6)
        return phi

    def get_unnormalized_phi(self, state, action):
        input = torch.cat([state, action], dim=1)
        phi = F.relu(self.feature_l1(input))
        phi = F.relu(self.feature_l2(phi))
        phi = F.relu(self.feature_l3(phi))
        # phi = spectral_norm(phi, norm_bound=0.95, n_power_iterations=1)

        return phi

    def forward(self, state, action):
        phi = self.get_phi(state, action)
        Q = self.weight_l1(phi)
        return Q

    def get_reward(self, state, action):
        phi = self.get_phi(state, action)
        R = self.reward_l1(phi)
        return R

    def get_transition(self, state, action):
        phi = self.get_phi(state, action)
        pred = F.relu(self.forward_l1(phi))
        pred = F.relu(self.forward_l2(pred))
        pred = F.relu(self.forward_l3(pred))
        pred = self.forward_l4(pred)
        return pred

    def get_psi(self, state, action):
        phi = self.get_unnormalized_phi(state, action)
        phi = phi.detach()
        psi = F.relu(self.psi_l1(phi))
        psi = F.relu(self.psi_l2(psi))
        return psi


# sf and sac or dqn. for now, it is based on sac, only the q-value or ppo?
# must be off-policy, then sac or ddpg -> directly actor successor feature?
# get psi and w, how to update psi, how to update w
class Trainer:
    def __init__(self, num_inputs, action_space, args, model_type='shared'):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.sf = SharedSF(num_inputs, action_space.shape[0], args.feat_size, args.hidden_size).to(self.device)
        self.target_sf = SharedSF(num_inputs, action_space.shape[0], args.feat_size, args.hidden_size).to(self.device)
        self.sf_optim = Adam(self.sf.parameters(), lr=args.sf_lr)

        hard_update(self.target_sf, self.sf)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.model_type = model_type


    def to(self):
        for network in self.networks:
            network.to(self.device)

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.target_sf(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.sf(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # todo: get qf1 and qf2, minimize them
        if self.model_type == 'idp':
            '''
            loss for idp_SF
            '''
            phi = self.sf.get_phi(state_batch, action_batch) # fixed
            w = self.sf.get_w(state_batch, action_batch)
            psi = self.sf.get_psi(state_batch, action_batch)

            w_loss = torch.mean(torch.pow(reward_batch - torch.sum((phi.detach() * w), dim=1, keepdim=True), 2))  # only train w
            target_psi = self.target_sf.get_psi(next_state_batch, next_state_action)
            target_psi = (phi + self.gamma * mask_batch * target_psi).detach()
            psi_loss = torch.mean(torch.pow((target_psi - psi), 2))  # used to train psi,
        elif self.model_type == 'shared':
            '''
            loss for SharedSF
            '''
            phi = self.sf.local_embedding(state_batch, action_batch) # fixed
            weight, bias = self.sf.get_w(state_batch, action_batch)
            psi = self.sf.global_embedding(state_batch, action_batch)

            w_loss = F.smooth_l1_loss(F.linear(phi, weight, bias), reward_batch)
            target_psi = self.target_sf.global_embedding(next_state_batch, next_state_action)
            target_psi = (phi + self.gamma * mask_batch * target_psi).detach()
            psi_loss = torch.mean(torch.pow((target_psi - psi), 2))  # used to train psi,

        else:
            raise NotImplementedError

        # todo: constrastive
        sf_loss = qf_loss + w_loss + psi_loss

        # todo: how to get the feature of the phi
        # todo: maintain two copies of w and phi is okay?
        # todo: update w and ps separately
        self.sf_optim.zero_grad()
        sf_loss.backward()
        self.sf_optim.step()

        '''
        update policy network
        '''
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.sf(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.target_sf, self.sf, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    @property
    def networks(self):
        network_list = [self.sf, self.target_sf, self.policy]
        return network_list

        # train omega
        # phi is the input of the psi network, then ->



        # train theta, but theta or psi

    # how to build the successor feature network, how to train them
    # 1. reward training
    # 2. dqn training

