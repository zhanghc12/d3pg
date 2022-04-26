import numpy as np
import torch
import gym
import argparse
import os
import os.path as osp
import duelingpg.utils as utils

from duelingpg import dnpg_eval_v3

import datetime
from torch.utils.tensorboard import SummaryWriter


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    avg_len = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            avg_len += 1

    avg_reward /= eval_episodes
    avg_len /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}, episode length: {:.2f}".format(eval_episodes, avg_reward, avg_len))
    print("---------------------------------------")
    return avg_reward


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

    policy = dnpg_eval_v3.D3PG(**kwargs)

    stds = []
    if args.env == 'HalfCheetah-v2':
        stds = np.array([0.0267,0.0483,0.2409,0.2429,0.2939,0.2280,0.2310,0.1762,0.6867,0.5823,1.2553,7.9273,7.7772,8.2787,6.0047,7.6029,5.6397])
    elif args.env == 'Hopper-v2':
        stds = np.array([0.0397,0.0108,0.0925,0.0134,0.0608,0.2227,0.2087,0.3554,0.4259,0.4380,0.8584])
    elif args.env == 'Walker2d-v2':
        stds = np.array([0.0281,0.0521,0.0478,0.0716,0.0975,0.0553,0.0568,0.1018,0.2401,0.2233,2.2874,3.0197,2.4364,3.8227,3.1578,2.4878,3.8398])
    else:
        stds = np.array([0.0266,0.0266,0.0284,0.0256,0.0250,0.0814,0.1084,0.1069,0.0765,0.1257,0.0701,0.1237,0.0989,0.2565,0.2636,0.4562,1.1361,1.1066,0.7945,1.6805,2.3335,2.1514,1.6336,2.5113,1.4211,2.5081,2.0492,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1763,0.1758,0.0393,0.3257,0.3248,0.1781,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1047,0.1080,0.1020,0.1309,0.1314,0.1434,0.2785,0.2821,0.2746,0.3027,0.3009,0.3011,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0768,0.0751,0.0723,0.0932,0.0940,0.0973,0.2974,0.2912,0.2955,0.3071,0.3084,0.3120,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0760,0.0791,0.0734,0.0957,0.0948,0.0992,0.2846,0.2883,0.2877,0.2972,0.2996,0.2980,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1079,0.1073,0.1035,0.1329,0.1328,0.1450,0.2780,0.2802,0.2779,0.2984,0.2926,0.2994])


    replay_buffer = utils.PerturbedReplayBuffer(state_dim, action_dim, stds, target_threshold=args.target_threshold)
    onpolicy_buffer = utils.PerturbedReplayBuffer(state_dim, action_dim, stds, target_threshold=0)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        fake_done_bool = float(done) if episode_timesteps < env._max_episode_steps else 1


        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool, fake_done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            actor_loss, critic_loss, q1, q2, q_diff, bias_loss, bias_diff,  r2, r3 = policy.train(replay_buffer, args.batch_size)

            writer.add_scalar('loss/actor_loss', actor_loss, t)
            writer.add_scalar('loss/critic_loss', critic_loss, t)
            writer.add_scalar('value/q1', q1, t)
            writer.add_scalar('value/q2', q2, t)
            writer.add_scalar('value/q_diff', q_diff, t)
            writer.add_scalar('bias/bias_loss', bias_loss, t)
            writer.add_scalar('bias/bias_diff', bias_diff, t)


        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print(
            #    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            writer.add_scalar('train/return', episode_reward, t)

            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_return = eval_policy(policy, args.env, args.seed)
            evaluations.append(avg_return)
            writer.add_scalar('test/return', avg_return, t)
            # np.save(f"./results/{file_name}", evaluations)


