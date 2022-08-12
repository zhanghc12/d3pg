import duelingpg.utils as utils
import datetime
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch
import gym
import argparse
import os
from knn_ue import td3_aaai
import d4rl.gym_mujoco
import numpy as np
from sklearn.neighbors import KDTree
import time
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print("Steps:{}, Evaluation over {} episodes: {:.3f}, bc_scale:{}, normalized scoare:{}".format(t, eval_episodes, avg_reward, bc_scale, d4rl_score))
    print("---------------------------------------")
    return avg_reward, d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="sf")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
    parser.add_argument("--env", default="hopper-medium-replay-v0")  # OpenAI gym environment name
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
    parser.add_argument("--eta", default=0.1, type=float)

    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop", default=200, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--bc_scale", type=float, default=0.5)
    parser.add_argument("--loading", type=int, default=0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--is_random", type=int, default=1)
    parser.add_argument("--load", type=int, default=0)


    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/kd/'
        model_dir = '/data/zhanghc/kd/model/'

    else:
        experiment_dir = '/tmp/data/zhanghc/kd/'

    if not os.path.exists(experiment_dir):
        experiment_dir = '/DATA/disk1/zhanghc/kd/'
        model_dir = '/DATA/disk1/zhanghc/kd/model/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    experiment_dir = experiment_dir + '0810/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_eta{}_tau{}_d{}_n{}_bs{}_od{}_k{}_r{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.eta, args.tau, args.top_quantiles_to_drop, args.n_nets, args.bc_scale, args.output_dim, args.k, args.is_random))

    file_name = args.policy + "_"  + args.env + "_"  + str(args.seed)
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    drop_quantile_bc = 0
    policy = td3_aaai.TD3(state_dim, action_dim, args.discount, args.tau, args.bc_scale, args.eta, args.n_nets, args.n_quantiles, args.top_quantiles_to_drop, drop_quantile_bc, args.output_dim, args.version, args.k, args.is_random)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    offline_dataset = d4rl.qlearning_dataset(env)
    obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

    # load predefined data or not
    if args.is_random == 0:
        if not args.load:
            for i in trange(100000):
                policy.train_feature_extractor(replay_buffer, batch_size=256)
            torch.save(policy.feature_nn.state_dict(), model_dir + "vae" + "_" + args.env)

        else:
            policy.feature_nn.load_state_dict(torch.load(model_dir + "vae" + "_" + args.env))

    # Evaluate untrained policy
    evaluations = [eval_policy(0, policy, args.env, args.seed, obs_mean, obs_std, args.bc_scale)]

    kdtree_path = experiment_dir + 'kdtree/critic' + args.env
    iid_list_path = experiment_dir + 'kdtree/iid_list' + args.env

    print(kdtree_path)

    trees = []

    data = np.concatenate([replay_buffer.state, replay_buffer.action], axis=1)
    if not torch.cuda.is_available():
        data = data[:10000]
    print('start build tree')
    i = 0
    batch_size = 2560
    phi_list = []
    i = 0
    while i + batch_size < replay_buffer.size:
        print(i)
        index = np.arange(i, i + batch_size)
        state_batch, action_batch = replay_buffer.sample_by_index(ind=index)
        phi = policy.feature_nn(state_batch, action_batch)
        phi_list.extend(phi.detach().cpu().numpy())
        i += batch_size
    index = np.arange(i, replay_buffer.size)
    state_batch, action_batch = replay_buffer.sample_by_index(ind=index)
    phi = policy.feature_nn(state_batch, action_batch)
    phi_list.extend(phi.detach().cpu().numpy())

    trees = KDTree(np.array(phi_list), leaf_size=40)

    start_time = time.time()

    for t in range(int(args.max_timesteps)):
        critic_loss, actor_loss = policy.train_policy(replay_buffer, args.batch_size, trees)

        if t % 100 == 0:
            writer.add_scalar('loss/critic_loss', critic_loss, t)
            writer.add_scalar('loss/actor_loss', actor_loss, t)
            print('iteration: {}, critic_loss :{:4f}, actor_loss: {:4f}, left_time:{:.2f}'.format(t, critic_loss, actor_loss, (time.time() - start_time) / 100 * (1e6 - t) / 3600 ))
            start_time = time.time()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_return, d4rl_score = eval_policy(t, policy, args.env, args.seed, obs_mean, obs_std, args.bc_scale)
            evaluations.append(avg_return)
            writer.add_scalar('test/return', avg_return, t)
            writer.add_scalar('test/d4rl_score', d4rl_score, t)
