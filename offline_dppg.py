import duelingpg.utils as utils
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import gym
import argparse
from knn_ue import dppg
import d4rl.gym_mujoco
import numpy as np
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data - mean) / (std + 1e-5)

def load_hdf5(dataset, replay_buffer):
    obs_mean = np.mean(dataset['observations'], axis=0, keepdims=True)
    obs_std = np.std(dataset['observations'], axis=0, keepdims=True)

    states = normalize(dataset['observations'])
    actions = dataset['actions']
    next_states = normalize(dataset['next_observations'])
    rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    dones = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    next_actions = np.concatenate([dataset['actions'][1:], dataset['actions'][-1:]], axis=0)

    for state, action, reward, next_state, done, next_action in tqdm(zip(states, actions, rewards, next_states, dones, next_actions)):
        replay_buffer.add(state, action, reward, next_state, done, next_action)

    print('loading finished!!!')
    return obs_mean, obs_std

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(t, policy, env_name, seed, obs_mean, obs_std, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (state - obs_mean) / (obs_std + 1e-5)
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print("Steps:{}, Evaluation over {} episodes: {:.3f}, normalized scoare:{}".format(t, eval_episodes, avg_reward, d4rl_score))
    print("---------------------------------------")
    return avg_reward, d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="sf")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
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

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bc_scale", type=float, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/offline_dppg/'
    else:
        experiment_dir = '/tmp/data/zhanghc/offline_dppg/'
    experiment_dir = experiment_dir + '0609/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}_alpha{}_bc{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau, args.alpha, args.bc_scale))

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

    policy = dppg.TD3(state_dim, action_dim, args.discount, args.tau, args.bc_scale)

    replay_buffer = utils.PrioritizedReplayBuffer(size=int(2e6), alpha=args.alpha)
    offline_dataset = d4rl.qlearning_dataset(env)
    obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

    # Evaluate untrained policy
    evaluations = [eval_policy(0, policy, args.env, args.seed, obs_mean, obs_std)]

    start_time = time.time()

    for t in range(int(args.max_timesteps)):
        critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)

        if t % 100 == 0:
            writer.add_scalar('loss/critic_loss', critic_loss, t)
            writer.add_scalar('loss/actor_loss', actor_loss, t)
            print('iteration: {}, critic_loss :{:4f}, actor_loss: {:4f}, left_time:{:.2f}'.format(t, critic_loss, actor_loss, (time.time() - start_time) / 100 * (1e6 - t) / 3600 ))
            start_time = time.time()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_return, d4rl_score = eval_policy(t, policy, args.env, args.seed, obs_mean, obs_std)
            evaluations.append(avg_return)
            writer.add_scalar('test/return', avg_return, t)
            writer.add_scalar('test/d4rl_score', d4rl_score, t)
