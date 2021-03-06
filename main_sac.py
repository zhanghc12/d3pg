import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac.sac import SAC
from sac.sac_dueling import DuelingSAC
from torch.utils.tensorboard import SummaryWriter
from sac.replay_memory import ReplayMemory


def eval_policy(policy, env_name, seed, eval_episodes=10):
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
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--version', type=int, default=5,
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--target_threshold', type=float, default=0., metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--model_version', type=int, default=1,
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--target_version', type=int, default=0,
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--policy_version', type=int, default=0,
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

eval_env = gym.make(args.env_name)
eval_env.seed(args.seed)
eval_env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
if args.version == 0:
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
else:
    agent = DuelingSAC(env.observation_space.shape[0], env.action_space, args)


if torch.cuda.is_available():
    experiment_dir = '/data/zhanghc/d3pg/'
    args.start_steps = 1000
else:
    experiment_dir = '/tmp/data/zhanghc/d3pg/'
    args.start_steps = 1000

experiment_dir = experiment_dir + '10_27/'
writer = SummaryWriter(
    experiment_dir + 'DSAC_{}_{}_s{}_ver{}_thre{}_mv{}_tv{}_pv{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.seed, args.version, args.target_threshold, args.model_version, args.target_version, args.policy_version))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            action, behavior_log_prob = agent.select_action(state, return_log_prob=True)
        else:
            action, behavior_log_prob = agent.select_action(state, return_log_prob=True)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, importance_ratio, importance_ratio_max, importance_ratio_min = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                writer.add_scalar('signal/importance_ratio', importance_ratio, updates)
                writer.add_scalar('signal/importance_ratio_max', importance_ratio_max, updates)
                writer.add_scalar('signal/importance_ratio_min', importance_ratio_min, updates)


                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask, behavior_log_prob) # Append transition to memory

        state = next_state

        if total_numsteps % 5000 == 0 and args.eval is True:
            eval_avg_reward = 0.
            eval_episodes = 10
            for _ in range(eval_episodes):
                eval_state = eval_env.reset()
                eval_episode_reward = 0
                eval_done = False
                while not eval_done:
                    eval_action = agent.select_action(eval_state, evaluate=True)

                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_episode_reward += eval_reward

                    eval_state = eval_next_state
                eval_avg_reward += eval_episode_reward
            eval_avg_reward /= eval_episodes

            writer.add_scalar('avg_reward/test', eval_avg_reward, total_numsteps)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(total_numsteps, round(eval_avg_reward, 2)))
            print("----------------------------------------")

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))


env.close()