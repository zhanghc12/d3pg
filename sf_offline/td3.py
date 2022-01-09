import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sf_offline.utils import hard_update
from sf_offline.successor_feature import IdpSF, Actor
import copy


class TD3(object):
    def __init__(self, state_dim, action_dim, gamma, tau, bc_scale):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sf = IdpSF(state_dim=state_dim, action_dim=action_dim, feat_dim=256, hidden_dim=256).to(self.device)
        self.sf_optim = Adam(self.sf.parameters(), lr=3e-4)
        self.sf_target = IdpSF(state_dim=state_dim, action_dim=action_dim, feat_dim=256, hidden_dim=256).to(self.device)
        hard_update(self.sf_target, self.sf)

        self.policy = Actor(state_dim, action_dim, 1).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)
        self.policy_target = copy.deepcopy(self.policy)

        self.bc_scale = bc_scale

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.policy(state)
        return action.detach().cpu().numpy()[0]

    def train_reward(self, memory, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size)
        predict_reward = self.sf.forward_reward(state_batch, action_batch, fix_feature=False)
        reward_loss = F.mse_loss(predict_reward, reward_batch)
        self.sf_optim.zero_grad()
        reward_loss.backward()
        self.sf_optim.step()
        # assume that the feature is randomized
        return reward_loss.item()

    def train_policy(self, memory, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size)

        # policy evaluation
        # todo: for target network, the phi is fixed, we only update psi network, done
        # todo: this is td3 network, there is no entropy, done
        with torch.no_grad():
            next_state_action = self.policy_target(next_state_batch)
            psi_next_target = self.sf_target.get_psi(next_state_batch, next_state_action)
            phi_state_action = self.sf_target.get_phi(state_batch, action_batch)
            psi_next_target = phi_state_action + mask_batch * self.gamma * (psi_next_target)
            psi_next_target = psi_next_target.detach()
        psi_state_action = self.sf.get_psi(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        psi_loss = torch.mean((psi_state_action - psi_next_target) ** 2)
        self.sf_optim.zero_grad()
        psi_loss.backward()
        self.sf_optim.step()

        # policy improvement
        psi_state_action = self.sf.get_psi(state_batch, self.policy(state_batch))
        curr_Q = self.sf(state_batch, self.policy(state_batch))
        lmbda = 1 / (curr_Q.abs().mean() + 1e-2).detach()
        bc_loss = self.bc_scale / (psi_state_action.norm(dim=1, keepdim=True) + 1e-6)
        policy_loss = -lmbda * curr_Q.mean() + bc_loss.mean() # 1
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # update the target network
        for param, target_param in zip(self.sf.parameters(), self.sf_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return psi_loss.item(), -curr_Q.mean().item()

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

