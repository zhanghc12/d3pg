import duelingpg.utils as utils
import datetime
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch
import gym
import argparse
import os
from knn_ue import td3_final, knn_utils
import d4rl.gym_mujoco
import numpy as np
from sklearn.neighbors import KDTree


def normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data - mean) / (std + 1e-5)

def load_hdf5(dataset, replay_buffer):
    obs_mean = np.mean(dataset['observations'], axis=0, keepdims=True)
    obs_std = np.std(dataset['observations'], axis=0, keepdims=True)
    replay_buffer.state = normalize(dataset['observations'])
    replay_buffer.action = dataset['actions']
    replay_buffer.next_state = normalize(dataset['next_observations'])
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer.next_action = np.concatenate([dataset['actions'][1:],dataset['actions'][-1:]], axis=0)
    replay_buffer.forward_label = normalize(np.concatenate([dataset['next_observations'] - dataset['observations'], np.expand_dims(np.squeeze(dataset['rewards']), 1)], axis=1))

    replay_buffer.size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer.not_done.sum())
    return obs_mean, obs_std

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(t, policy, env_name, seed, obs_mean, obs_std, bc_scale, eval_episodes=10, bc=False):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (state - obs_mean) / (obs_std + 1e-5)
            action = policy.select_action(np.array(state), bc=bc)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Steps:{}, Evaluation over {} episodes: {:.3f}, bc_scale:{}".format(t, eval_episodes, avg_reward, bc_scale))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="sf")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
    #parser.add_argument("--env", default="hopper-medium-replay-v2")  # OpenAI gym environment name
    parser.add_argument("--env", default="halfcheetah-medium-replay-v0")  # OpenAI gym environment name

    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--version", default=0, type=int)
    parser.add_argument("--target_threshold", default=0.1, type=float)

    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop", default=200, type=int)
    parser.add_argument("--n_nets", default=10, type=int)
    parser.add_argument("--bc_scale", type=float, default=0.5)
    parser.add_argument("--loading", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)


    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/kd/'
    else:
        experiment_dir = '/tmp/data/zhanghc/kd/'

    envs_list = [
        'hopper-random-v0',
        'hopper-medium-v0',
        'hopper-medium-replay-v0',
        'hopper-medium-expert-v0',
        'hopper-expert-v0',
        'walker2d-random-v0',
        'walker2d-medium-v0',
        'walker2d-medium-replay-v0',
        'walker2d-medium-expert-v0',
        'walker2d-expert-v0',
        'halfcheetah-random-v0',
        'halfcheetah-medium-v0',
        'halfcheetah-medium-replay-v0',
        'halfcheetah-medium-expert-v0',
        'halfcheetah-expert-v0',
    ]

    envs_list = [
        'walker2d-medium-replay-v0', #.
    ]

    for env_name in envs_list:
        print(env_name)
        env = gym.make(env_name)

        # Set seeds
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
        offline_dataset = d4rl.qlearning_dataset(env)
        obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

        # Evaluate untrained policy

        data = np.concatenate([replay_buffer.state, replay_buffer.action], axis=1)
        print('start build tree')
        tree = KDTree(data, leaf_size=40)

        iid_list = knn_utils.test_tree_true(replay_buffer, tree, k=2)


