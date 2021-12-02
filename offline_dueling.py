import numpy as np
import torch
import gym
import argparse
import os
import os.path as osp
import duelingpg.utils as utils
from duelingpg import d3pg_offline
from duelingpg import OurDDPG
from duelingpg import dtpg
from duelingpg import dhpg
from duelingpg import d2pg
from duelingpg import d6pg
from duelingpg import d4pg
from duelingpg import dvpg
import d4rl
import datetime
from torch.utils.tensorboard import SummaryWriter




def load_hdf5(dataset, replay_buffer):
    replay_buffer.state = dataset['observations']
    replay_buffer.action = dataset['actions']
    replay_buffer.next_state = dataset['next_observations']
    replay_buffer.reward = dataset['rewards']
    replay_buffer.not_done = 1 - dataset['terminals']


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
    parser.add_argument("--policy", default="Dueling")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
    parser.add_argument("--env", default="hopper-random-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
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
    parser.add_argument("--version", default=6, type=int)
    parser.add_argument("--target_threshold", default=0.1, type=float)

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/d3pg/'
    else:
        experiment_dir = '/tmp/data/zhanghc/d3pg/'
    experiment_dir = experiment_dir + '1202/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau))

    file_name = args.policy + "_"  + args.env + "_"  + str(args.seed)
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
    }

    # Initialize policy
    if args.policy == "Dueling":
        #kwargs["policy_noise"] = args.policy_noise * max_action
        #kwargs["noise_clip"] = args.noise_clip * max_action
        #kwargs["policy_freq"] = args.policy_freq
        kwargs['version'] = args.version
        kwargs['target_threshold'] = args.target_threshold

        policy = d3pg_offline.D3PG(**kwargs)
    elif args.policy == "D4PG":
        #kwargs["policy_noise"] = args.policy_noise * max_action
        #kwargs["noise_clip"] = args.noise_clip * max_action
        #kwargs["policy_freq"] = args.policy_freq
        kwargs['version'] = args.version
        kwargs['scale'] = args.target_threshold

        policy = d4pg.D4PG(**kwargs)

    elif args.policy == "DVPG":
        #kwargs["policy_noise"] = args.policy_noise * max_action
        #kwargs["noise_clip"] = args.noise_clip * max_action
        #kwargs["policy_freq"] = args.policy_freq
        kwargs['version'] = args.version
        # kwargs['scale'] = args.target_threshold

        policy = dvpg.DVPG(**kwargs)

    elif args.policy == "D6PG":
        #kwargs["policy_noise"] = args.policy_noise * max_action
        #kwargs["noise_clip"] = args.noise_clip * max_action
        #kwargs["policy_freq"] = args.policy_freq
        kwargs['version'] = args.version
        kwargs['target_threshold'] = args.target_threshold

        policy = d6pg.D6PG(**kwargs)
    elif args.policy == 'DTPG':
        kwargs['version'] = args.version
        policy = dtpg.DTPG(**kwargs)
    elif args.policy == 'DHPG':
        kwargs['version'] = args.version
        policy = dhpg.DHPG(**kwargs)
    elif args.policy == 'DMPG':
        # kwargs['version'] = args.version
        kwargs['env_name'] = args.env
        kwargs['version'] = args.version
        policy = dmpg.DMPG(**kwargs)
    elif args.policy == 'DRPG':
        # kwargs['version'] = args.version
        kwargs['env_name'] = args.env
        kwargs['version'] = args.version
        policy = drpg.DRPG(**kwargs)
    elif args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "D2PG":
        policy = d2pg.D2PG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    #if args.load_model != "":
    #    policy_file = file_name if args.load_model == "default" else args.load_model
    #    policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    onpolicy_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e4))

    offline_dataset = d4rl.qlearning_dataset(env)
    load_hdf5(offline_dataset, replay_buffer)

    # Evaluate untrained policy
    evaluations = [eval_policy(0, policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        actor_loss, critic_loss, weight_loss, pi_Q_mean, pi_Q_max, pi_Q_min, current_Q_mean, current_Q_max, current_Q_min, \
           current_adv_mean, current_adv_max, current_adv_min, \
           current_value_mean, current_value_max, current_value_min, \
           target_v_mean, target_v_max, target_v_min = policy.train(replay_buffer, args.batch_size)

        writer.add_scalar('loss/actor_loss', actor_loss, t)
        writer.add_scalar('loss/critic_loss', critic_loss, t)
        writer.add_scalar('loss/weight_loss', weight_loss, t)
        writer.add_scalar('value/current_Q_mean', current_Q_mean, t)
        writer.add_scalar('value/current_Q_max', current_Q_max, t)
        writer.add_scalar('value/current_Q_min', current_Q_min, t)

        writer.add_scalar('value/pi_Q_mean', pi_Q_mean, t)
        writer.add_scalar('value/pi_Q_max', pi_Q_max, t)
        writer.add_scalar('value/pi_Q_min', pi_Q_min, t)

        writer.add_scalar('value/current_adv_mean', current_adv_mean, t)
        writer.add_scalar('value/current_adv_max', current_adv_max, t)
        writer.add_scalar('value/current_adv_min', current_adv_min, t)

        writer.add_scalar('value/current_value_mean', current_value_mean, t)
        writer.add_scalar('value/current_value_max', current_value_max, t)
        writer.add_scalar('value/current_value_min', current_value_min, t)

        writer.add_scalar('value/target_v_mean', target_v_mean, t)
        writer.add_scalar('value/target_v_max', target_v_max, t)
        writer.add_scalar('value/target_v_min', target_v_min, t)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_return = eval_policy(t, policy, args.env, args.seed)
            evaluations.append(avg_return)
            writer.add_scalar('test/return', avg_return, t)
            # np.save(f"./results/{file_name}", evaluations)
            # if args.save_model: policy.save(f"./models/{file_name}")