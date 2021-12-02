import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.etd3 import SACNTrainer
from rlkit.torch.core import eval_np

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch.nn as nn
import torch.nn.functional as F

import torch
import argparse, os
import numpy as np
import os
import os.path
import shutil
import h5py
import d4rl, gym
import datetime


def load_hdf5(dataset, replay_buffer):
    if isinstance(replay_buffer, EnvReplayBuffer):
        replay_buffer._observations = dataset['observations']
        replay_buffer._next_obs = dataset['next_observations']
        replay_buffer._actions = dataset['actions']
        replay_buffer._next_actions = dataset['actions']

        replay_buffer._mc_returns = np.expand_dims(np.squeeze(dataset['rewards']), 1)
        replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
        replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
        replay_buffer._remain_steps = np.expand_dims(np.squeeze(dataset['rewards']), 1)
        replay_buffer._random_percent = np.expand_dims(np.squeeze(dataset['rewards']), 1)  # new 1
        replay_buffer._n_random = 3

        replay_buffer._indexes = np.expand_dims(np.squeeze(dataset['rewards']), 1)

        replay_buffer._size = dataset['terminals'].shape[0]
        print('Number of terminals on: ', replay_buffer._terminals.sum())
        replay_buffer._top = replay_buffer._size

    mean_obs = np.mean(dataset['observations'], axis=0)
    std_obs = np.std(dataset['observations'], axis=0)

    mean_actions = np.mean(dataset['actions'], axis=0)
    std_actions = np.std(dataset['actions'], axis=0)

    mean_rewards = np.mean(dataset['rewards'], axis=0)
    std_rewards = np.std(dataset['rewards'], axis=0)

    if True:
        std_obs[std_obs < 1e-5] = 1e-5
    return mean_obs, std_obs, mean_actions, std_actions, mean_rewards, std_rewards


class DuelingCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingCritic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.lv = nn.Linear(256, 1)

        self.l3 = nn.Linear(256 + action_dim, 256)
        self.la = nn.Linear(256, 1)

    def forward(self, state, action):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value = self.lv(feat)
        adv = F.relu(self.l3(torch.cat([feat, action], 1)))
        adv = self.la(adv)
        return value, adv, value + adv

    def forward_adv(self, state, action):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        feat = feat.detach()
        adv = F.relu(self.l3(torch.cat([feat, action], 1)))
        adv = self.la(adv)
        return adv

    def get_value(self, state):
        feat = F.relu(self.l2(F.relu(self.l1(state))))
        value = self.lv(feat)
        return value


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state, deterministic=True):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions, {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]


def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']

    critics = nn.ModuleList()
    target_critics = nn.ModuleList()

    for i in range(2):
        critics.append(DuelingCritic(
            state_dim=obs_dim,
            action_dim=action_dim))

    for i in range(2):
        target_critics.append(DuelingCritic(
            state_dim=obs_dim,
            action_dim=action_dim))

    policy = Actor(
        state_dim=obs_dim,
        action_dim=action_dim,
        max_action=1
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    if variant['load_buffer'] and buffer_filename is not None:
        offline_dataset = None
        replay_buffer.load_buffer(buffer_filename)
    elif 'random-expert' in variant['env_name']:
        offline_dataset = d4rl.mixed_dataset(eval_env, random_percent=variant['random_percent'], full_expert=False)
        mean_obs, std_obs, mean_actions, std_actions, mean_rewards, std_rewards = load_hdf5(
            offline_dataset, replay_buffer)
    elif 'mix-expert' in variant['env_name']:
        offline_dataset = d4rl.mixed_dataset(eval_env, random_percent=variant['random_percent'],
                                             full_expert=False, medium=True)
        mean_obs, std_obs, mean_actions, std_actions, mean_rewards, std_rewards = load_hdf5(
            offline_dataset, replay_buffer)

    else:
        offline_dataset = d4rl.qlearning_dataset(eval_env)
        mean_obs, std_obs, mean_actions, std_actions, mean_rewards, std_rewards = load_hdf5(
            offline_dataset, replay_buffer)

    trainer = SACNTrainer(
        env=eval_env,
        policy=policy,
        critics=critics,
        target_critics=target_critics,

        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        summary_log_dir=variant['summary_log_dir'],
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],

        is_syn_training=False,
        is_random_ordered=True,
        is_random_split=False,
        is_random_average=False,
        is_save_eval_path=False,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str is not "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        summary_log_dir='./',
        random_percent=0,

        algorithm_kwargs=dict(
            num_epochs=1100,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=5.0,

            # lagrange
            with_lagrange=True,  # Defaults to true
            lagrange_thresh=10.0,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,

            count_threshold=0.1,
            base_threshold=0.0,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper-random-v0')
    parser.add_argument("--gpu", default='', type=str)
    parser.add_argument("--max_q_backup", type=str,
                        default="False")  # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str,
                        default="True")  # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=10000,
                        type=int)  # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=5.0,
                        type=float)  # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)  # Policy learning rate
    parser.add_argument('--min_q_version', default=1, type=int)  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
    parser.add_argument('--lagrange_thresh', default=5.0,
                        type=float)  # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--random_percent', default=0.9, type=float)

    # arguments for training model
    parser.add_argument("--count_threshold", type=float, default=0.1)
    parser.add_argument("--base_threshold", type=float, default=0.)

    args = parser.parse_args()

    enable_gpus(args.gpu)
    if 'random-expert' in args.env:
        post_env = args.env + '-' + str(args.random_percent)
    elif 'mix-expert' in args.env:
        post_env = args.env + '-' + str(args.random_percent)
    else:
        post_env = args.env

    if torch.cuda.is_available():
        exp_dir = os.path.join('/data/zhanghc/d4rl/sacn/', post_env)
        log_dir = '/data/zhanghc/d4rl/sacn/'
    else:
        exp_dir = os.path.join('/tmp/data/zhanghc/d4rl/sacn', post_env)
        log_dir = '/tmp/data/zhanghc/d4rl/sacn/'

    variant['random_percent'] = args.random_percent

    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh

    variant['summary_log_dir'] = os.path.join(log_dir, 'tf_logs_1123', post_env,
                                              str(variant['trainer_kwargs']['min_q_version']),
                                              'thre-' + str(args.count_threshold) + '-bthre' + str(
                                                  args.base_threshold) + '-lag' + str(
                                                  variant['trainer_kwargs']['lagrange_thresh']) + '-w' + str(
                                                  variant['trainer_kwargs']['min_q_weight']),
                                              datetime.datetime.now().strftime('%c'), str(args.seed))

    variant['buffer_filename'] = None
    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed
    variant['post_env'] = post_env

    setup_logger('local_logs/', variant=variant, base_log_dir=exp_dir)

    if torch.cuda.is_available():
        ptu.set_gpu_mode(True)
    experiment(variant)
