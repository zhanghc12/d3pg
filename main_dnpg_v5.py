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

    model_dir = '/data/zhanghc/tf_models/' + args.env + '/'
    env_model = construct_model(obs_dim=state_dim, act_dim=action_dim, hidden_dim=200,
                                num_networks=7,
                                num_elites=5, model_dir=model_dir, name='BNN_115000')
    predict_env = PredictEnv(env_model, args.env, 'tensorflow')


    policy = dnpg_eval_v4.D3PG(**kwargs)

    replay_buffer = utils.ModeledReplayBuffer(state_dim, action_dim, target_threshold=args.target_threshold)
    onpolicy_buffer = utils.ModeledReplayBuffer(state_dim, action_dim, target_threshold=0)

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
        perturbed_next_state, perturbed_reward = predict_env.step_single(state, action)
        perturbed_next_state = next_state + args.target_threshold * (perturbed_next_state - next_state)
        perturbed_reward = reward + args.target_threshold *(perturbed_reward - reward)
        if args.version == 100:
            perturbed_reward = reward

        # 0, no noise
        # 1, pure perturb
        # 0.01
        # 0.1
        '''
        if t % 100 == 0:
            print('state_error', np.abs(perturbed_next_state - next_state))
            print('reward_error', np.abs(reward - perturbed_reward))
            print('rel_state_error', np.abs(perturbed_next_state - next_state) / (np.abs(next_state) + 1e-6))

            # 0.11055  0.88451  0.45427  0.34102  0.34408  0.42239  0.50313  0.33358
            #    3.77046  0.67164  2.14682  9.21376  7.98837  7.55857  9.19066 11.35497
            #    7.65146  0.80107  0.66895  0.72724  0.61391  0.76352  0.65932

            # [[-4.70140e-02  2.33330e-01  3.74848e-02 -1.78363e-01 -1.51083e-02                           
            #   -5.78601e-02 -5.55856e-03 -1.83146e-02  8.36824e+00 -6.51337e-02                           
            #    7.18851e-04  2.12533e-01 -3.98862e-02  7.87675e-02 -2.08685e-01                           
            #    3.89809e-01  2.27202e-01 -6.12492e-03 -2.63280e-01 -7.85856e-02                           
            #   -8.56887e-02 -5.51485e-02  9.98859e-03]]                                                   
            # [[ 0.10939  0.87077  0.45566  0.34358  0.3448   0.42436  0.50425  0.33355                    
            #    3.7581   0.67351  2.15283  9.25037  8.04784  7.56236  9.24105 11.38824                    
            #    7.66265  0.80172  0.66845  0.72792  0.61433  0.76279  0.6594 ]] 
        '''
        replay_buffer.add(state, action, next_state, reward, done_bool, fake_done_bool, perturbed_next_state, perturbed_reward)


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


