import numpy as np
import torch
import gym
import argparse
import os
import os.path as osp
import duelingpg.utils as utils
import shutil
from duelingpg import dnpg_eval_v2

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
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
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
    parser.add_argument("--first_phase", default=1, type=int)

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/dnpg/'
    else:
        experiment_dir = '/tmp/data/zhanghc/dnpg/'
    experiment_dir = experiment_dir + '04_12/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}_n{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau, args.num_critic))

    # file_name = "{}_{}_{}".format(args.policy, args.env, args.seed)
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")

    #if not os.path.exists("./results"):
    #    os.makedirs("./results")

    #if args.save_model and not os.path.exists("./models"):
    #    os.makedirs("./models")

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
        "num_critic": args.num_critic
    }

    # Initialize policy

    kwargs['version'] = args.version
    kwargs['target_threshold'] = args.target_threshold

    policy = dnpg_eval_v2.D3PG(**kwargs)

    dirname = experiment_dir + 'env{}_target{}'.format(args.env, args.target_threshold) + '/'
    filename = dirname + 'models'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif os.path.exists(dirname) and args.first_phase == 1:
        shutil.rmtree(dirname)
        os.makedirs(dirname)
    elif os.path.exists(dirname) and args.first_phase == 2:
        policy.load_policy(filename)
        print('loaded!')

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    onpolicy_buffer = utils.ReplayBuffer(state_dim, action_dim)

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
            if args.first_phase == 1:
                actor_loss, critic_loss, q1, q2, q_value, target_qvalue, r1,  r2, r3 = policy.train_first_phase(replay_buffer, args.batch_size)
            else:
                actor_loss, critic_loss, q1, q2, q_value, target_qvalue, r1,  r2, r3 = policy.train_second_phase(replay_buffer, args.batch_size)

            writer.add_scalar('loss/actor_loss', actor_loss, t)
            writer.add_scalar('loss/critic_loss', critic_loss, t)
            writer.add_scalar('value/q1', q1, t)
            writer.add_scalar('value/q2', q2, t)
            #writer.add_scalar('value/q_value', q_value, t)
            #writer.add_scalar('value/target_qvalue', target_qvalue, t)
            #writer.add_scalar('value/r1', r1, t)
            #writer.add_scalar('value/r2', r2, t)
            #writer.add_scalar('value/r3', r3, t)

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


        if torch.cuda.is_available():
            if (t + 1) % 1000000 == 0 and args.test == 1 and args.first_phase == 2:
                final_rewards, final_states, final_actions = utils.test_mc_v3(args.env, policy, onpolicy_buffer)
                value_eval, value_train, value_diff, value_ratio = policy.eval_value_clip(final_rewards, final_states, final_actions)
                writer.add_scalar('value/value_eval', value_eval, t)
                writer.add_scalar('value/value_train', value_train, t)
                writer.add_scalar('value/value_diff', value_diff, t)
                writer.add_scalar('value/value_ratio', value_ratio, t)
                # print('end', value_eval, value_train, value_diff, value_ratio)

        if t % 1000==0 and args.first_phase == 1:
            policy.save_policy(filename=filename)
