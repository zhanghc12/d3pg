import torch
import torch.nn as nn
import torch.nn.functional as F


'''
train the w and phi, get phi(s,a) and w(s,a), make sure phi(s,a) > 0
'''


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
        w = F.relu(self.weight_l1(input))
        w = F.relu(self.weight_l2(w))
        w = self.weight_l3(w)
        return w

# how to train the offline

