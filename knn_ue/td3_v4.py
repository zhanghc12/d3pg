import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sf_offline.utils import hard_update
from sf_offline.successor_feature import IdpSF, Actor, Critic, MixedSF, SepSF
import copy
import numpy as np


class TD3(object):
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.policy = Actor(state_dim, action_dim, 1).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)
        self.policy_target = copy.deepcopy(self.policy)

        self.bc_scale = bc_scale
        # still td3, but do two other thing, estimate psi(s,a)

        self.bc_critic = SepSF(state_dim=state_dim, action_dim=action_dim, feat_dim=256, hidden_dim=256).to(self.device)
        self.bc_critic_optim = Adam(self.bc_critic.parameters(), lr=3e-4)
        self.bc_critic_target = SepSF(state_dim=state_dim, action_dim=action_dim, feat_dim=256, hidden_dim=256).to(self.device)
        hard_update(self.bc_critic_target, self.bc_critic)
        self.bc_policy = Actor(state_dim, action_dim, 1).to(self.device)
        self.bc_policy_optim = Adam(self.bc_policy.parameters(), lr=3e-4)

        self.total_it = 0

    def select_action(self, state, evaluate=False, bc=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if bc:
            action = self.bc_policy(state)
        else:
            action = self.policy(state)
        return action.detach().cpu().numpy()[0]

    def train_bc(self, memory, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch, next_action_batch, forward_label = memory.sample_v3(batch_size, include_next_action=True)

        # reward loss
        # predict_reward = self.bc_critic.get_reward(state_batch, action_batch)
        # reward_loss = F.mse_loss(predict_reward, reward_batch)

        # transiton loss
        pred = self.bc_critic.get_transition(state_batch, action_batch)
        reward_loss = torch.mean((pred - forward_label)** 2)

        # psi loss
        with torch.no_grad():
            psi_next_target = self.bc_critic_target.get_psi(next_state_batch, next_action_batch)
            phi_state_action = self.bc_critic.get_phi(state_batch, action_batch)
            psi_next_target = (phi_state_action + mask_batch * self.gamma * psi_next_target).detach()
            q_next_target = self.bc_critic(next_state_batch, next_action_batch)
            q_next_target = (reward_batch + mask_batch * self.gamma * q_next_target).detach()

        curr_psi = self.bc_critic.get_psi(state_batch, action_batch)
        psi_loss = torch.mean((curr_psi - psi_next_target) ** 2)
        curr_q = self.bc_critic(state_batch, action_batch)
        q_loss = F.mse_loss(curr_q, q_next_target)

        ood_action1 = action_batch + 1 * torch.normal(torch.zeros_like(action_batch), torch.ones_like(action_batch))
        ood_action2 = action_batch + 0.1 * torch.normal(torch.zeros_like(action_batch), torch.ones_like(action_batch))
        ood_action = torch.cat([ood_action1, ood_action2], dim=0)
        ood_state = torch.cat([state_batch, state_batch], dim=0)
        ood_loss = self.bc_critic.get_psi(ood_state, ood_action)
        ood_loss = torch.mean(ood_loss.norm(dim=-1, p=1))

        total_loss = 0.001 * reward_loss + 1000 * psi_loss + 0.1 * ood_loss
        self.bc_critic_optim.zero_grad()
        total_loss.backward()
        self.bc_critic_optim.step()

        curr_Q = self.bc_critic(state_batch, self.bc_policy(state_batch))
        policy_loss = -curr_Q.mean()
        self.bc_policy_optim.zero_grad()
        policy_loss.backward()
        self.bc_policy_optim.step()

        # there is no policy to update
        for param, target_param in zip(self.bc_critic.parameters(), self.bc_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        iid_psi = self.bc_critic.get_psi(state_batch, action_batch)
        iid_psi = torch.mean(torch.norm(iid_psi, dim=1, p=1))

        # ood_psi = self.bc_critic.get_psi(torch.normal(torch.zeros_like(state_batch), torch.ones_like(state_batch)),  torch.normal(torch.zeros_like(action_batch), torch.ones_like(action_batch)))
        ood_psi = self.bc_critic.get_psi(state_batch,  torch.clamp_(action_batch + 0.1 * torch.normal(torch.zeros_like(action_batch), torch.ones_like(action_batch)), -1, 1))

        # ood_psi = self.bc_critic.get_psi( torch.normal(torch.zeros_like(action_batch), torch.ones_like(action_batch)), torch.clamp_(action_batch + 10 * torch.normal(torch.zeros_like(action_batch), torch.ones_like(action_batch)), -1, 1))
        ood_psi = torch.mean(torch.norm(ood_psi, dim=1, p=1))

        return reward_loss, psi_loss, q_loss, ood_loss, iid_psi.item(), ood_psi.item()

    def get_stat(self, mean_distance):
        self.mean_distance = mean_distance

    def get_stat_test(self, memory, batch_size=256):
        i = 0
        self.psi_list = []
        while i + batch_size < memory.size:
            index = np.arange(i, i+batch_size)
            state_batch, action_batch = memory.sample_by_index(ind=index)
            psi = self.bc_critic.get_psi(state_batch, action_batch)
            psi_norm = psi.norm(dim=1, p=1)
            i += batch_size
            self.psi_list.extend(psi_norm.detach().cpu().numpy())
        self.min_psi_norm = np.min(self.psi_list)
        partion_num = np.int32((memory.size * self.bc_scale))
        self.partion_psi_norm = np.array(self.psi_list)[np.argpartition(self.psi_list, partion_num)][partion_num]
        self.max_psi_norm = np.max(self.psi_list)


    def train_policy(self, memory, batch_size, kd_tree):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size)
        with torch.no_grad():
            next_action = self.policy_target(next_state_batch)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + mask_batch * self.gamma * target_Q

            #next_state_batch_np = state_batch.cpu().numpy()
            #next_action_batch_np = next_action.detach().cpu().numpy()
            #query_data = np.concatenate([next_state_batch_np, next_action_batch_np], axis=1)
            #target_distance = kd_tree.query(query_data, k=3)[0]
            #target_distance = np.mean(target_distance, axis=1, keepdims=True)
            #target_flag = torch.FloatTensor(target_distance < self.mean_distance).to(self.device)
            #target_psi = self.bc_critic.get_psi(next_state_batch, next_action)
            #target_psi_norm = target_psi.norm(dim=1, keepdim=True, p=1)
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)

        # todo: penalize the target by the psi norm

        # Compute critic loss
        if self.total_it % 5000 == 0:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        else:
            critic_loss = 0 * F.mse_loss(current_Q1, target_Q) + 0 * F.mse_loss(current_Q2, target_Q)
        # critic_loss = torch.mean(target_flag * (current_Q1 - target_Q) ** 2) + torch.mean(target_flag * (current_Q2 - target_Q) ** 2)

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic.Q1(state_batch, self.policy(state_batch)).mean() - self.critic.Q2(state_batch,
                                                                                                    self.policy(
                                                                                                        state_batch)).mean()
        # Delayed policy updates
        if self.total_it % 2 == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state_batch, self.policy(state_batch)).mean() -self.critic.Q2(state_batch, self.policy(state_batch)).mean()

            # Optimize the actor
            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1
        return critic_loss.item(), actor_loss.item()


    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

