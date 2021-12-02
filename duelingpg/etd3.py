from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd


class SACNTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            critics,
            target_critics,
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs=2,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,

            count_threshold=0,
            base_threshold=0
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.critics = critics
        self.target_critics = target_critics
        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.critic_optimizer = optimizer_class(
            self.critics.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        self.num_qs = num_qs

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        # For implementation on the
        self.discrete = False

        self.count_threshold = count_threshold
        self.base_threshold = base_threshold

        ptu.soft_update_from_to(
            self.critics, self.target_critics, 1.
        )

    def train_from_torch(self, batch, batch_uniform):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        self.num_critic = len(self.critics)
        start_epoch = 0

        if self._current_epoch <= start_epoch:
            pi = self.policy(obs)
            policy_loss = ((pi - actions) ** 2).mean()
        else:
            advs = []
            pi_action = self.policy(obs)
            for i in range(self.num_critic):
                advs.append(self.critics[i](obs, pi_action)[-1])  # -2 to -1

            advs = torch.squeeze(torch.cat(advs, dim=-1))
            policy_loss = -torch.mean(advs, dim=-1)  # [0]
            policy_loss = (1 - self.count_threshold) * policy_loss.mean() + self.count_threshold * (
                        (pi_action - actions) ** 2).mean()

        if self._current_epoch % 2 == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        QF Loss
        """

        '''

        rewards = batch_uniform['rewards']
        terminals = batch_uniform['terminals']
        obs = batch_uniform['observations']
        actions = batch_uniform['actions']
        next_obs = batch_uniform['next_observations']
        '''

        target_vs = []
        for i in range(self.num_critic):
            target_vs.append(self.target_critics[i].get_value(next_obs))

        target_q_values = torch.mean(torch.squeeze(torch.cat(target_vs, dim=-1)), dim=-1, keepdim=True)  # [0]
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        qf_loss = 0.
        alpha_prime_loss = 0.
        for i in range(self.num_critic):
            v_pred, adv_pred, q_pred = self.critics[i](obs, actions)
            _, adv_pi, _ = self.critics[i](obs, self.policy(obs))
            # todo: define mask, the i*bs: (i+1)*bs is 1, other is zeros
            # masks = torch.zeros_like(rewards.to(ptu.device))
            # masks[i * 256: (i+1) * 256] = 1
            # qf_loss += (masks * (q_pred - adv_pi - q_target) ** 2).mean()
            qf_loss += self.qf_criterion(q_pred - adv_pi, q_target)

            if self.min_q_version == 1:
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                adv_gap = (torch.abs(adv_pred - adv_pi)).mean()
                qf_loss += alpha_prime[0] * (adv_gap)

                alpha_prime_loss += -alpha_prime[0] * (adv_gap - self.target_action_gap).detach()

        """
        Update networks
        """
        # Update the Q-functions iff
        self._num_q_update_steps += 1
        if self._current_epoch > start_epoch:
            if self.min_q_version == 1:
                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            self.critic_optimizer.zero_grad()
            qf_loss.backward()
            self.critic_optimizer.step()

        self._num_policy_update_steps += 1

        """
        Soft Updates
        """
        if self._current_epoch % 2 == 0:
            ptu.soft_update_from_to(
                self.critics, self.target_critics, self.soft_target_tau
            )

            """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf_loss))

            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'actions',
                    ptu.get_numpy(actions)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'rewards',
                    ptu.get_numpy(rewards)
                ))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

            v_pred, adv_pred, q_pred = self.critics[0](obs, actions)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q_pred),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.critics,
            self.target_critics,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            critics=self.critics,
            target_critics=self.target_critics,
        )

