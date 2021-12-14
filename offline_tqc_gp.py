import numpy as np
import torch
import gym
import argparse
import os
import os.path as osp
import duelingpg.utils as utils
from duelingpg import d3pg_offline
from duelingpg import d3pg_offline_1
from duelingpg import d3pg_offline_2
from duelingpg import d3pg_offline_3

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

import numpy as np
import torch
import gym
import argparse
import os
from tqc import tqc_offline
import gpytorch
from tqc.gp_models import MultitaskGPModel
from torch.distributions import Normal
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_assess(model, likelihood, X, Y):
    lik, sq_diff = [], []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X))
        m = observed_pred.mean
        v = observed_pred.variance
    avg_v = torch.mean(v)

    Y = Y.cpu().data.numpy()
    m = m.cpu().data.numpy()
    v = v.cpu().data.numpy()

    l = np.sum(norm.logpdf(Y, loc=m, scale=v ** 0.5), 1)
    sq = ((m - Y) ** 2)

    lik.append(l)
    sq_diff.append(sq)

    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    return np.average(lik), np.average(sq_diff) ** 0.5, avg_v



def load_hdf5(dataset, replay_buffer):
    replay_buffer.state = dataset['observations']
    replay_buffer.action = dataset['actions']
    replay_buffer.next_state = dataset['next_observations']
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)


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
    parser.add_argument("--policy", default="tqc")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
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
    parser.add_argument("--version", default=3, type=int)
    parser.add_argument("--target_threshold", default=0.1, type=float)

    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=248, type=int)
    parser.add_argument("--n_nets", default=10, type=int)
    parser.add_argument('--gp_rank', help='Rank of the task covar module', type=int, default=1)
    parser.add_argument('--kernel_type', help='Kernel for the GP', type=str, default='rbf')

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/d3pg/'
    else:
        experiment_dir = '/tmp/data/zhanghc/d3pg/'
    experiment_dir = experiment_dir + '1202/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}_d{}_n{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau, args.top_quantiles_to_drop_per_net, args.n_nets))

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
    kwargs['version'] = args.version
    kwargs['target_threshold'] = args.target_threshold
    kwargs['top_quantiles_to_drop_per_net'] = args.top_quantiles_to_drop_per_net
    kwargs['n_nets'] = args.n_nets
    policy = tqc_offline.TQC(**kwargs)


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

    train_x = torch.from_numpy(offline_dataset['observations'][:1000]).float().to(device)
    train_y = torch.from_numpy(offline_dataset['actions'][:1000]).float().to(device)

    # Initialize likelihood and model
    gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=action_dim).to(device)

    gp_model = MultitaskGPModel(train_x, train_y, gp_likelihood, num_tasks=action_dim, rank=args.gp_rank,
                             ard_num_dims=state_dim, kernel_type=args.kernel_type).to(device)

    gp_model.train()
    gp_likelihood.train()

    epochs = 1000

    trainloader = torch.utils.data.DataLoader(train_x, shuffle=True, batch_size=64)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likelihood, gp_model)
    gp_optimizer = torch.optim.Adam([
        {'params': gp_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    for epoch in range(epochs):
        gp_optimizer.zero_grad()
        output = gp_model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        gp_optimizer.step()

        trainl, trains, avg_v = batch_assess(gp_model, gp_likelihood, train_x, train_y)


        print('Iter %d/%d - Loss: %.3f, Train mean log likelihood: %.3f, Train RMSE: %.3f' % (
            epoch, epochs, loss.item(), trainl, trains
        ))

        '''
        for batch_obs in trainloader:



            
            # Minimise KL - just consider the observations.
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = gp_likelihood(gp_model(batch_obs))
                nm = Normal(observed_pred.mean, observed_pred.variance)
            
            #prior_log_prob = nm.log_prob(action).sum(dim=1, keepdim=True)
            #policy_loss = (log_pi - prior_log_prob).mean()
        '''

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        actor_loss, critic_loss, top_quantiles_to_drop, normalized_std_z_iod, normalized_std_z_ood = policy.train(replay_buffer, args.batch_size)

        writer.add_scalar('loss/actor_loss', actor_loss, t)
        writer.add_scalar('loss/critic_loss', critic_loss, t)
        writer.add_scalar('loss/top_quantiles_to_drop', top_quantiles_to_drop, t)
        writer.add_scalar('loss/normalized_std_z_iod', normalized_std_z_iod, t)
        writer.add_scalar('loss/normalized_std_z_ood', normalized_std_z_ood, t)


        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_return = eval_policy(t, policy, args.env, args.seed)
            evaluations.append(avg_return)
            writer.add_scalar('test/return', avg_return, t)
            # np.save(f"./results/{file_name}", evaluations)
            # if args.save_model: policy.save(f"./models/{file_name}")



