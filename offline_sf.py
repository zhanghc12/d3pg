import duelingpg.utils as utils
import d4rl
import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import gym
import argparse
import os
from sf_offline import td3
import d4rl.gym_mujoco


def load_hdf5(dataset, replay_buffer):
    replay_buffer.state = dataset['observations']
    replay_buffer.action = dataset['actions']
    replay_buffer.next_state = dataset['next_observations']
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer.next_action = np.concatenate([dataset['actions'][1:],dataset['actions'][-1]], axis=0)


    replay_buffer.size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer.not_done.sum())

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(t, policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Steps:{}, Evaluation over {} episodes: {:.3f}".format(t, eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="sf")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
    parser.add_argument("--env", default="hopper-random-v0")  # OpenAI gym environment name
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
    parser.add_argument("--version", default=3, type=int)
    parser.add_argument("--target_threshold", default=0.1, type=float)

    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=248, type=int)
    parser.add_argument("--n_nets", default=10, type=int)
    parser.add_argument("--bc_scale", type=float, default=0.5)


    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/sf/'
    else:
        experiment_dir = '/tmp/data/zhanghc/sf/'
    experiment_dir = experiment_dir + '0108/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}_d{}_n{}_bs{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau, args.top_quantiles_to_drop_per_net, args.n_nets, args.bc_scale))

    file_name = args.policy + "_"  + args.env + "_"  + str(args.seed)
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")

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

    policy = td3.TD3(state_dim, action_dim, args.discount, args.tau, args.bc_scale)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    offline_dataset = d4rl.qlearning_dataset(env)
    load_hdf5(offline_dataset, replay_buffer)

    # Evaluate untrained policy
    evaluations = [eval_policy(0, policy, args.env, args.seed)]

    #  first, get a fixed weight, but do we need to add spectral normalization to this layer?
    for t in range(int(args.max_timesteps / 10)):
        phi_loss = policy.train_reward(replay_buffer, args.batch_size)  # todo 1: feature collapse, spectral nomalization
        if t % 100 == 0:
            print('iteration', t, phi_loss)
            writer.add_scalar('loss/phi_loss', phi_loss, t)


    for t in range(int(args.max_timesteps)):
        psi_loss, policy_loss = policy.train_policy(replay_buffer, args.batch_size)

        if t % 100 == 0:
            writer.add_scalar('loss/psi_loss', psi_loss, t)
            writer.add_scalar('loss/policy_loss', policy_loss, t)
            print('iteration: {}, policy loss :{:6f}, psi loss: {:6f}'.format(t, policy_loss, psi_loss))

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_return = eval_policy(t, policy, args.env, args.seed)
            evaluations.append(avg_return)
            writer.add_scalar('test/return', avg_return, t)
            # np.save(f"./results/{file_name}", evaluations)
            # if args.save_model: policy.save(f"./models/{file_name}")