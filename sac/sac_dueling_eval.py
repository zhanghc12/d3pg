import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy, DuelingNetworkv0, DuelingNetworkv1, DuelingNetworkv2
from sac.autoregressive_policy_v1 import AutoRegressiveStochasticActor

class DuelingSAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.version = args.version
        self.model_version = args.model_version
        self.target_version = args.target_version
        self.policy_version = args.policy_version

        if self.model_version == 0:
            self.critic = DuelingNetworkv0(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = DuelingNetworkv0(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.critic_eval = DuelingNetworkv0(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        elif self.model_version == 1:
            self.critic = DuelingNetworkv1(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = DuelingNetworkv1(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.critic_eval = DuelingNetworkv1(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        elif self.model_version == 2:
            self.critic = DuelingNetworkv2(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = DuelingNetworkv2(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.critic_eval = DuelingNetworkv2(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

        self.critic_eval_optim = Adam(self.critic_eval.parameters(), lr=args.lr)

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

        if self.policy_version == 0:
            self.behavior_policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        elif self.policy_version == 1:
            self.behavior_policy = AutoRegressiveStochasticActor(num_inputs, action_space.shape[0]).to(self.device)
        self.behavior_policy_optim = Adam(self.behavior_policy.parameters(), lr=args.lr)



    def select_action(self, state, evaluate=False, return_log_prob=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, log_prob, _ = self.policy.sample(state)
        else:
            _, log_prob, action = self.policy.sample(state)
        if return_log_prob:
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0]

    def update_eval_parameters(self, memory, updates, batch_size):
        if updates % 1000 == 0:
            state_batch, action_batch, return_batch = memory.sample(
                batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            return_batch = torch.FloatTensor(return_batch).to(self.device).unsqueeze(1)


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, behavior_log_prob_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        behavior_log_prob_batch = torch.FloatTensor(behavior_log_prob_batch).to(self.device)

        '''
        update Q function
        '''
        with torch.no_grad():
            vf1_next_target, vf2_next_target = self.critic_target.get_value(next_state_batch)
            if self.target_version == 0:
                min_vf_next_target = torch.min(vf1_next_target, vf2_next_target) # under estimate, value is not accurate enough,
            elif self.target_version == 1:
                min_vf_next_target = (vf1_next_target + vf2_next_target) / 2# under estimate, value is not accurate enough
            elif self.target_version == 2:
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                adv1_next_target, adv2_next_target = self.critic_target.get_adv(next_state_batch, next_state_action)
                if self.version == 1:
                    _, _, target_pi_1 = self.policy.sample(state_batch)
                    target_adv_pi_1, target_adv_pi_2 = self.critic_target.get_adv(state_batch, target_pi_1)
                if self.version in [2]:
                    self.num_repeat = 100
                    target_state_bath_temp = next_state_batch.unsqueeze(1).repeat(1, self.num_repeat, 1).view(
                        next_state_batch.shape[0] * self.num_repeat, next_state_batch.shape[1])
                    target_pi_temp, _, _ = self.policy.sample(target_state_bath_temp)
                    target_adv_pi_1, target_adv_pi_2 = self.critic_target.get_adv(target_state_bath_temp, target_pi_temp)
                    target_adv_pi_1 = target_adv_pi_1.view(state_batch.shape[0], self.num_repeat, 1)
                    target_adv_pi_1 = target_adv_pi_1.mean(dim=1)
                    target_adv_pi_2 = target_adv_pi_2.view(state_batch.shape[0], self.num_repeat, 1)
                    target_adv_pi_2 = target_adv_pi_2.mean(dim=1)

                target_entropy = self.policy.get_entropy(next_state_batch).detach()

                vf1_next_target = vf1_next_target + adv1_next_target - target_adv_pi_1 - self.alpha * target_entropy
                vf2_next_target = vf2_next_target + adv2_next_target - target_adv_pi_2 - self.alpha * target_entropy

                min_vf_next_target = torch.min(vf1_next_target, vf2_next_target) - self.alpha * next_state_log_pi # under estimate, value is not accurate enough,

            next_q_value = reward_batch + mask_batch * self.gamma * (min_vf_next_target)  # todo: min -> mean -> variance
            '''
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            '''
        value_1, adv_1, qf1, value_2, adv_2, qf2 = self.critic(state_batch, action_batch, return_full=True)

        if self.version in [1,3,4,5]:
            _, _, pi_1 = self.policy.sample(state_batch)
            adv_pi_1, adv_pi_2 = self.critic.get_adv(state_batch, pi_1)
        if self.version in [2]:
            self.num_repeat = 100
            state_bath_temp = state_batch.unsqueeze(1).repeat(1, self.num_repeat, 1).view(state_batch.shape[0] * self.num_repeat, state_batch.shape[1])
            pi_temp, _, _ = self.policy.sample(state_bath_temp)
            adv_pi_1, adv_pi_2 = self.critic.get_adv(state_bath_temp, pi_temp)
            adv_pi_1 = adv_pi_1.view(state_batch.shape[0], self.num_repeat, 1)
            adv_pi_1 = adv_pi_1.mean(dim=1)
            adv_pi_2 = adv_pi_2.view(state_batch.shape[0], self.num_repeat, 1)
            adv_pi_2 = adv_pi_2.mean(dim=1)

        # todo: enumerate all the samples
        log_prob = self.policy.log_prob(state_batch, action_batch).detach()
        entropy = self.policy.get_entropy(state_batch).detach()

        qf1_loss = F.mse_loss(value_1 + adv_1 - (adv_pi_1 + self.alpha * entropy) , next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(value_2 + adv_2 - (adv_pi_2 + self.alpha * entropy), next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #qf1_loss = F.mse_loss(qf1 , next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #qf2_loss = F.mse_loss(qf2 , next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]


        qf_loss = qf1_loss + qf2_loss
        '''
        update value function
        '''
        # v_t = E_pi(r + gamma V(s_t+1))
        if self.version in [3, 4, 5] and updates >= 1000:
            with torch.no_grad():
                if self.version == 5:
                    behavior_log_prob = behavior_log_prob_batch
                else:
                    behavior_log_prob = self.behavior_policy.log_prob(state_batch, action_batch)

                log_prob = self.policy.log_prob(state_batch, action_batch)
                vf1_next_target, vf2_next_target = self.critic_target.get_value(next_state_batch)
                min_vf_next_target = torch.min(vf1_next_target, vf2_next_target)
                entropy = self.alpha * self.policy.get_entropy(state_batch).detach()
                next_v = reward_batch + mask_batch * self.gamma * min_vf_next_target  # todo, we need to get entroph here
                importance_ratio = (log_prob - behavior_log_prob).exp()
                # normalized_importance_ratio = importance_ratio.clamp_(0.,10)

                # importance_ratio = importance_ratio / (importance_ratio.sum() + 1e-2)
                normalized_importance_ratio = torch.clamp(importance_ratio, 0, 10.) # 0.5 0.01 0.1
                # normalized_importance_ratio = importance_ratio.clamp_(0., 3)

                #normalized_importance_ratio = normalized_importance_ratio.clamp_(0.1, 10)
                next_v = normalized_importance_ratio * next_v

            vf1, vf2 = self.critic.get_value(state_batch)

            #vf1_loss = ((normalized_importance_ratio * (vf1 - next_v)) ** 2).mean()
            #vf2_loss = ((normalized_importance_ratio * (vf2 - next_v)) ** 2).mean()

            vf1_loss = F.mse_loss(vf1 - entropy, next_v)
            vf2_loss = F.mse_loss(vf2 - entropy, next_v)
            vf_loss = vf1_loss + vf2_loss
            qf_loss = vf_loss + qf_loss
        else:
            importance_ratio = torch.tensor(0.)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        '''
        update behavior policy
        '''
        if self.version == 3:
            behavior_log_prob = self.behavior_policy.log_prob(state_batch, action_batch)
            behavior_policy_loss = -behavior_log_prob.mean()
            self.behavior_policy_optim.zero_grad()
            behavior_policy_loss.backward()
            self.behavior_policy_optim.step()

        if self.version == 4:
            extended_batch_size = min(batch_size*10, len(memory.buffer))
            extended_state_batch, extended_action_batch, extended_reward_batch, extended_next_state_batch, extended_mask_batch = memory.sample(batch_size=extended_batch_size)
            extended_state_batch = torch.FloatTensor(extended_state_batch).to(self.device)
            extended_action_batch = torch.FloatTensor(extended_action_batch).to(self.device)

            behavior_log_prob = self.behavior_policy.log_prob(extended_state_batch, extended_action_batch)
            behavior_policy_loss = -behavior_log_prob.mean()
            self.behavior_policy_optim.zero_grad()
            behavior_policy_loss.backward()
            self.behavior_policy_optim.step()
        '''
        update actor
        '''
        #pi, log_pi, _ = self.policy.sample(state_batch)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)

        if self.version in [1, 3, 4, 5]:
            qf1_pi, qf2_pi = self.critic.get_adv(state_batch, pi)

            _, _, pi_1 = self.policy.sample(state_batch)
            adv_pi_1, adv_pi_2 = self.critic.get_adv(state_batch, pi_1)
            qf1_pi = qf1_pi - adv_pi_1.detach()
            qf2_pi = qf2_pi - adv_pi_2.detach()
        if self.version in [2]:
            qf1_pi, qf2_pi = self.critic.get_adv(state_batch, pi)
            self.num_repeat = 100
            state_bath_temp = state_batch.unsqueeze(1).repeat(1, self.num_repeat, 1).view(state_batch.shape[0] * self.num_repeat, state_batch.shape[1])
            pi_temp, _, _ = self.policy.sample(state_bath_temp)
            adv_pi_1, adv_pi_2 = self.critic.get_adv(state_bath_temp, pi_temp)
            adv_pi_1 = adv_pi_1.view(state_batch.shape[0], self.num_repeat, 1)
            adv_pi_1 = adv_pi_1.mean(dim=1)
            adv_pi_2 = adv_pi_2.view(state_batch.shape[0], self.num_repeat, 1)
            adv_pi_2 = adv_pi_2.mean(dim=1)

            qf1_pi = qf1_pi - adv_pi_1.detach()
            qf2_pi = qf2_pi - adv_pi_2.detach()

        if self.target_version == 0:
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
        else:
            min_qf_pi = (qf1_pi + qf2_pi) / 2


        policy_loss = (self.alpha*log_pi- min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]  # todo: min_advantage ?

        #policy_loss = (2 * self.alpha*log_pi1).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]  # todo: min_advantage ?

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            # i, log_pi, _ = self.policy.sample(state_batch)

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_tlogs = self.alpha# .clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha, importance_ratio.mean().item(), importance_ratio.max().item(), importance_ratio.min().item()

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

