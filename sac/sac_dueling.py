import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy, DuelingNetwork


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
        self.critic = DuelingNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = DuelingNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

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

        self.behavior_policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.behavior_policy_optim = Adam(self.behavior_policy.parameters(), lr=args.lr)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        '''
        update Q function
        '''
        with torch.no_grad():
            vf1_next_target, vf2_next_target = self.critic_target.get_value(next_state_batch)
            min_vf_next_target = (vf1_next_target + vf2_next_target) / 2 # under estimate, value is not accurate enough,
            next_q_value = reward_batch + mask_batch * self.gamma * (min_vf_next_target)  # todo: min -> mean -> variance
            '''
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            '''
        value_1, adv_1, qf1, value_2, adv_2, qf2 = self.critic(state_batch, action_batch, return_full=True)

        if self.version == 1:
            _, _, pi_1 = self.policy.sample(state_batch)
            adv_pi_1, adv_pi_2 = self.critic.get_adv(state_batch, pi_1)
        if self.version in [2, 3]:
            self.num_repeat = 20
            state_bath_temp = state_batch.unsqueeze(1).repeat(1, self.num_repeat, 1).view(state_batch.shape[0] * self.num_repeat, state_batch.shape[1])
            pi_temp, _, _ = self.policy.sample(state_bath_temp)
            adv_pi_1, adv_pi_2 = self.critic.get_adv(state_bath_temp, pi_temp)
            adv_pi_1 = adv_pi_1.view(state_batch.shape[0], self.num_repeat, 1)
            adv_pi_1 = adv_pi_1.mean(dim=1)
            adv_pi_2 = adv_pi_2.view(state_batch.shape[0], self.num_repeat, 1)
            adv_pi_2 = adv_pi_2.mean(dim=1)

        # todo: enumerate all the samples
        log_prob = self.policy.log_prob(state_batch, action_batch).detach()

        qf1_loss = F.mse_loss(qf1 - adv_pi_1 + self.alpha * log_prob, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2 - adv_pi_2 + self.alpha * log_prob, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #qf1_loss = F.mse_loss(qf1 , next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        #qf2_loss = F.mse_loss(qf2 , next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]


        qf_loss = qf1_loss + qf2_loss
        '''
        update value function
        '''
        # v_t = E_pi(r + gamma V(s_t+1))
        if self.version == 3  and updates >= 10000:
            with torch.no_grad():
                behavior_log_prob = self.behavior_policy.log_prob(state_batch, action_batch)
                log_prob = self.policy.log_prob(state_batch, action_batch)
                vf1_next_target, vf2_next_target = self.critic_target.get_value(next_state_batch)
                min_vf_next_target = (vf1_next_target + vf2_next_target) / 2
                next_v = reward_batch + mask_batch * self.gamma * min_vf_next_target - self.alpha * log_prob  # todo, we need to get entroph here
                importance_ratio = (log_prob - behavior_log_prob).exp()
                normalized_importance_ratio = importance_ratio.clamp_(0.1,10)
                #normalized_importance_ratio = importance_ratio / importance_ratio.sum()
                #normalized_importance_ratio = normalized_importance_ratio.clamp_(0.1, 10)
                next_v = normalized_importance_ratio * next_v

            vf1, vf2 = self.critic.get_value(state_batch)

            vf1_loss = F.mse_loss(vf1, next_v)
            vf2_loss = F.mse_loss(vf2, next_v)
            vf_loss = vf1_loss + vf2_loss
            qf_loss = vf_loss + qf_loss

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

        '''
        update actor
        '''
        #pi, log_pi, _ = self.policy.sample(state_batch)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = (qf1_pi +  qf2_pi) / 2

        policy_loss = (- min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]  # todo: min_advantage ?

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

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha

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

