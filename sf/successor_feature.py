import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sf.utils import soft_update, hard_update
from sf.model import GaussianPolicy


class SuccessorFeature(nn.Module):
    def __init__(self, state_dim, action_dim, feat_dim, hidden_dim):
        super(SuccessorFeature, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim

        self.weight_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.weight_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.weight_l3 = nn.Linear(hidden_dim, self.feat_dim) # w : 1 * feat_dim

        self.phi_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.phi_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.phi_l3 = nn.Linear(hidden_dim, self.feat_dim) # phi: 1 * feat_dim

        self.psi_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.psi_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.psi_l3 = nn.Linear(hidden_dim, self.feat_dim) # psi: 1 * feat_dim

    def forward(self, state, action):
        # get successor feature of (state, action) pair: w(s,a): reward
        input = torch.cat([state, action], dim=1)
        w = F.relu(self.weight_l1(input))
        w = F.relu(self.weight_l2(w))
        w = self.weight_l3(w)

        psi = F.relu(self.psi_l1(input))
        psi = F.relu(self.psi_l2(psi))
        psi = self.psi_l3(psi)

        Q = (w * psi).sum(dim=1, keepdim=True)
        return Q

    def get_phi(self, state, action):
        input = torch.cat([state, action], dim=1)
        phi = F.relu(self.phi_l1(input))
        phi = F.relu(self.phi_l2(phi))
        phi = self.phi_l3(phi)
        return phi

    def get_psi(self, state, action):
        input = torch.cat([state, action], dim=1)
        psi = F.relu(self.l4(input))
        psi = F.relu(self.l5(psi))
        psi = self.l6(psi)
        return psi

    def get_w(self, state, action):
        input = torch.cat([state, action], dim=1)
        w = F.relu(self.l1(input))
        w = F.relu(self.l2(w))
        w = self.l3(w)
        return w


# sf and sac or dqn. for now, it is based on sac, only the q-value or ppo?
# must be off-policy, then sac or ddpg -> directly actor successor feature?
# get psi and w, how to update psi, how to update w
class Trainer:
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.sf = SuccessorFeature(num_inputs, action_space.shape[0], args.feat_size, args.hidden_size).to(self.device)
        self.target_sf = SuccessorFeature(num_inputs, action_space.shape[0], args.feat_size, args.hidden_size).to(self.device)
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

        # to set w
        phi = self.sf.get_phi(state_batch, action_batch) # fixed
        w = self.sf.get_w(state_batch, action_batch)
        psi = self.sf.get_psi(state_batch, action_batch)

        w_loss = torch.mean(torch.pow(reward_batch - phi.detach() * w, 2))  # only train w
        target_psi = self.target_sf.get_psi(next_state_batch, next_state_action)
        target_psi = (phi + self.gamma * mask_batch * target_psi).detach()
        psi_loss = torch.mean(torch.pow((target_psi - psi), 2))  # used to train psi,

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

