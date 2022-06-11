import numpy as np
import torch
import gym
import argparse
import os
import os.path as osp
import duelingpg.utils as utils

from duelingpg import dnpg_eval_v4

import datetime
from torch.utils.tensorboard import SummaryWriter

from mbpo_py.tf_models.constructor import construct_model, format_samples_for_training
from mbpo_py.predict_env import PredictEnv

import d4rl
import d4rl.gym_mujoco

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def load_hdf5(dataset, replay_buffer):
    obs_mean = np.mean(dataset['observations'], axis=0, keepdims=True)
    obs_std = np.std(dataset['observations'], axis=0, keepdims=True)
    replay_buffer.state = (dataset['observations'])
    # replay_buffer.action = normalize(dataset['actions'])
    replay_buffer.action = dataset['actions']

    replay_buffer.next_state = (dataset['next_observations'])
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer.next_action = np.concatenate([dataset['actions'][1:],dataset['actions'][-1:]], axis=0)
    replay_buffer.forward_label = (np.concatenate([dataset['next_observations'] - dataset['observations'], np.expand_dims(np.squeeze(dataset['rewards']), 1)], axis=1))

    replay_buffer.size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer.not_done.sum())
    return obs_mean, obs_std


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="D4PG")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
    parser.add_argument("--env", default="Ant-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--version", default=1, type=int)
    parser.add_argument("--target_threshold", default=0.1, type=float)
    parser.add_argument("--num_critic", default=2, type=int)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--ratio", default=0.1, type=float)


    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/dnpg/'
    else:
        experiment_dir = '/tmp/data/zhanghc/dnpg/'
    experiment_dir = experiment_dir + '04_12/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}_n{}_r{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau, args.num_critic, args.ratio))

    file_name = "{}_{}_{}".format(args.policy, args.env, args.seed)
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")

    #if not os.path.exists("./results"):
    #    os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "num_critic": args.num_critic,
        "ratio": args.ratio,
    }

    # Initialize policy

    kwargs['version'] = args.version
    kwargs['target_threshold'] = args.target_threshold

    model_dir = '/data/zhanghc/tf_models/' + args.env + '/'
    if args.version == 101:
        model_dir = ''

    env_model = construct_model(obs_dim=state_dim, act_dim=action_dim, hidden_dim=200,
                                num_networks=7,
                                num_elites=5, model_dir=model_dir, name='BNN_115000')
    predict_env = PredictEnv(env_model, args.env, 'tensorflow')
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    env = gym.make('hopper-expert-v2')
    offline_dataset = d4rl.qlearning_dataset(env)
    obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

    memory = replay_buffer
    i = 0
    batch_size = 256
    size = memory.size

    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_distance = predict_env.predict_uncertainty(state_batch, action_batch)
        i += batch_size
        print(i, np.mean(iid_distance))


