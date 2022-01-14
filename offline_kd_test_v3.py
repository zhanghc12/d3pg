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
import torch
import torch.nn.functional as F
import copy
import numpy as np
import torch.nn as nn
from torch.distributions import Distribution, Normal
from sf_offline.successor_feature import Actor
from tqc.spectral_normalization import spectral_norm
import math
import time

def normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data - mean) / (std + 1e-5)

def load_hdf5(dataset, replay_buffer):
    obs_mean = np.mean(dataset['observations'], axis=0, keepdims=True)
    obs_std = np.std(dataset['observations'], axis=0, keepdims=True)
    replay_buffer.state = normalize(dataset['observations'])
    # replay_buffer.action = normalize(dataset['actions'])
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



class ResBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, feat_dim=3):
        super(FeatureExtractor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim
        norm_bound = 10
        n_power_iterations = 1

        # first layer feature
        self.feature_l1 = spectral_norm(nn.Linear(self.state_dim + self.action_dim, hidden_dim), norm_bound=norm_bound,
                                        n_power_iterations=n_power_iterations)
        self.feature_l2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim), norm_bound=norm_bound, n_power_iterations=n_power_iterations)

        # self.feature_l25 = spectral_norm(nn.Linear(hidden_dim, hidden_dim), norm_bound=norm_bound, n_power_iterations=n_power_iterations)

        self.feature_l3 = spectral_norm(nn.Linear(hidden_dim, self.feat_dim), norm_bound=norm_bound,
                                        n_power_iterations=n_power_iterations)  # w : 1 * feat_dim

        # self.init_layers()
    def forward(self, state, action):
        # get successor feature of (state, action) pair: w(s,a): reward
        input = torch.cat([state, action], dim=1)
        # input = torch.cos(input)
        w = F.relu(self.feature_l1(input))
        # w = F.relu(self.feature_l2(w))
        w = F.relu(self.feature_l2(w))

        w = self.feature_l3(w)

        return w

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -0.03, 0.03)



class FeatureExtractorV4(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, feat_dim=3):
        super(FeatureExtractorV4, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim
        norm_bound = 10
        n_power_iterations = 1

        # first layer feature

        self.feature_l1 = nn.Sequential(

            nn.Linear(self.state_dim + self.action_dim, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )

        self.feature_l2 = nn.Linear(hidden_dim, self.feat_dim)

        # self.init_layers()
    def forward(self, state, action):
        # get successor feature of (state, action) pair: w(s,a): reward
        input = torch.cat([state, action], dim=1)
        # input = torch.cos(input)
        w = F.relu(self.feature_l1(input))
        # w = F.relu(self.feature_l2(w))
        # w = F.relu(self.feature_l2(w))

        w = self.feature_l2(w)

        return w

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -0.03, 0.03)

class FeatureExtractorV5(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, feat_dim=3):
        super(FeatureExtractorV5, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim
        norm_bound = 10
        n_power_iterations = 1

        # first layer feature

        self.feature_l1 = nn.Sequential(

            nn.Linear(self.state_dim + self.action_dim, 256),
            SpectralResBlock(256, 256, norm_bound),
            SpectralResBlock(256, 256, norm_bound),
        )

        self.feature_l2 = spectral_norm(nn.Linear(hidden_dim, self.feat_dim), norm_bound=norm_bound)

        # self.init_layers()
    def forward(self, state, action):
        # get successor feature of (state, action) pair: w(s,a): reward
        input = torch.cat([state, action], dim=1)
        # input = torch.cos(input)
        w = F.relu(self.feature_l1(input))
        # w = F.relu(self.feature_l2(w))
        # w = F.relu(self.feature_l2(w))

        w = self.feature_l2(w)

        return w

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -0.03, 0.03)



class FeatureExtractorV3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, feat_dim=3):
        super(FeatureExtractorV3, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = 1
        self.feat_dim = feat_dim
        norm_bound = 3
        n_power_iterations = 1

        # first layer feature
        self.feature_l1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.feature_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.feature_l3 = nn.Linear(hidden_dim, self.feat_dim)

        # self.feature_l25 = spectral_norm(nn.Linear(hidden_dim, hidden_dim), norm_bound=norm_bound, n_power_iterations=n_power_iterations)

        #self.feature_l3 = spectral_norm(nn.Linear(hidden_dim, self.feat_dim), norm_bound=norm_bound,
        #                                n_power_iterations=n_power_iterations)  # w : 1 * feat_dim

        # self.init_layers()
    def forward(self, state, action):
        # get successor feature of (state, action) pair: w(s,a): reward
        input = torch.cat([state, action], dim=1)
        w = F.relu(self.feature_l1(input))
        # w = F.relu(self.feature_l2(w))
        w = F.relu(self.feature_l2(w))

        w = self.feature_l3(w)

        return w

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -0.03, 0.03)

class ResBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class SpectralResBlock(nn.Module):

    def __init__(self, in_size, out_size, norm_bound):
        super(SpectralResBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Linear(in_size, out_size),norm_bound=norm_bound),
            nn.ReLU(),
            spectral_norm(nn.Linear(out_size, out_size),norm_bound=norm_bound),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class Head(nn.Module):
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super(Head, self).__init__()

        h_layer = 256
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
        norm_bound = 3

        #self.W1 = nn.Linear(h_layer, output_dim_in * output_dim_out)
        #self.b1 = nn.Linear(h_layer, output_dim_out)
        # self.s1 = nn.Linear(h_layer, output_dim_out)

        self.W1 = spectral_norm(nn.Linear(h_layer, output_dim_in * output_dim_out), norm_bound=norm_bound,
                      n_power_iterations=1)
        self.b1 = spectral_norm(nn.Linear(h_layer, output_dim_out), norm_bound=norm_bound,
                      n_power_iterations=1)
        self.s1 = spectral_norm(nn.Linear(h_layer, output_dim_out), norm_bound=norm_bound,
                      n_power_iterations=1)

        # self.init_layers(sttdev)

    def forward(self, x):
        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)
        s = 1. + self.s1(x).view(-1, self.output_dim_out, 1)

        return w, b, s

    def init_layers(self, stddev):
        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.s1.bias)
        torch.nn.init.zeros_(self.b1.bias)


class Meta_Embadding(nn.Module):
    def __init__(self, meta_dim, z_dim):
        super(Meta_Embadding, self).__init__()
        norm_bound = 0.95
        self.z_dim = z_dim
        self.hyper = nn.Sequential(
            spectral_norm(nn.Linear(meta_dim, 256),norm_bound=norm_bound,
                      n_power_iterations=1),
            nn.ReLU(),
            spectral_norm(nn.Linear(256, 256),norm_bound=norm_bound,
                      n_power_iterations=1),
            nn.ReLU(),
            #ResBlock(256, 256),
            #ResBlock(256, 256),
        )
        #spectral_norm(nn.Linear(self.state_dim + self.action_dim, hidden_dim), norm_bound=norm_bound, n_power_iterations=1)

        # self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -bound, bound)

class HyperNetwork(nn.Module): # s vs s to form a feature
    def __init__(self, meta_v_dim, base_v_dim, output_dim):
        super(HyperNetwork, self).__init__()

        dynamic_layer = 256
        z_dim = 256

        self.output_dim = output_dim

        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # Q function net
        self.layer1 = Head(z_dim, base_v_dim, dynamic_layer, sttdev=0.05)
        self.last_layer = Head(z_dim, dynamic_layer, output_dim, sttdev=0.008)

    def forward(self, meta_v, base_v, debug=None):
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1, b1, s1 = self.layer1(z)
        w2, b2, s2 = self.last_layer(z)

        # dynamic network pass
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = torch.bmm(w2, out) * s2 + b2

        return out.view(-1, self.output_dim)


class FeatureExtractorV2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, feat_dim=3):
        super(FeatureExtractorV2, self).__init__()
        self.hyper = HyperNetwork(state_dim,  action_dim, 3)
        #self.l1 = nn.Linear(256, 256)
        #self.l2 = nn.Linear(256, 256)
        #self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        inputs = torch.cat([state, action], 1)
        q = self.hyper(state, action)
        #q = F.relu(self.hyper(inputs, inputs))
        #q = F.relu(self.l1(q))
        #q = F.relu(self.l2(q))
        #return self.l3(q)
        return q


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
        'walker2d-medium-v0',        # 0.0033 0.03
        'walker2d-medium-replay-v0', # 0.0033, 0.03
        'walker2d-medium-expert-v0',
        'walker2d-expert-v0',
        'halfcheetah-random-v0',
        'halfcheetah-medium-v0',
        'halfcheetah-medium-replay-v0',
        'halfcheetah-medium-expert-v0',
        'halfcheetah-expert-v0',
    ]

    envs_list = [
        'halfcheetah-medium-v0', #.
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_nns = []
        phi_list = []
        size = 1
        for _ in range(size):
            feature_nns.append(FeatureExtractorV4(state_dim, action_dim, 256, 9).to(device))
            phi_list.append([])
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
        offline_dataset = d4rl.qlearning_dataset(env)
        obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

        # Evaluate untrained policy

        data = np.concatenate([replay_buffer.state, replay_buffer.action], axis=1)

        # save data in the replay buffer
        i = 0
        batch_size = 2560

        while i + batch_size < replay_buffer.size:
            print(i)
            index = np.arange(i, i + batch_size)
            state_batch, action_batch = replay_buffer.sample_by_index(ind=index)
            for j in range(size):
                phi = feature_nns[j](state_batch, action_batch)
                phi_list[j].extend(phi.detach().cpu().numpy())
            i += batch_size

        print('start build tree')
        trees = [KDTree(np.array(phi), leaf_size=40) for phi in phi_list]

        start_time = time.time()
        iid_list = knn_utils.test_tree_true_sns(replay_buffer, trees, feature_nns, k=2)
        print('eclipsd time:', time.time() - start_time)

