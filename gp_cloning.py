import d4rl
import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import gym
import argparse
import gpytorch
from tqc.gp_models import MultitaskGPModel
from torch.distributions import Normal
from scipy.stats import norm
import os
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rollout(
        env,
        model,
        likelihood,
        max_path_length=np.inf,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals
    The next element will be a list of dictionaries, with the index into
    the list being the index into the time
     - env_infos
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        while path_length < max_path_length:
            o_torch = torch.from_numpy(np.array([o])).float().to(device)
            observed_pred = likelihood(model(o_torch))
            a = observed_pred.mean.data.cpu().numpy()

            if len(a) == 1:
                a = a[0]

            next_o, r, d, env_info = env.step(a)

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )


def collect_new_paths(
        env,
        model,
        likelihood,
        max_path_length,
        num_steps,
        discard_incomplete_paths,
):
    paths = []
    num_steps_collected = 0
    while num_steps_collected < num_steps:
        max_path_length_this_loop = min(  # Do not go over num_steps
            max_path_length,
            num_steps - num_steps_collected,
        )

        path = rollout(
            env,
            model,
            likelihood,
            max_path_length=max_path_length_this_loop,
        )

        path_len = len(path['actions'])
        if (
                path_len != max_path_length
                and not path['terminals'][-1]
                and discard_incomplete_paths
        ):
            break
        num_steps_collected += path_len
        paths.append(path)
    return paths

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
    parser.add_argument('--evaluation_interval', help='Evaluation interval', type=int, default=20)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=248, type=int)
    parser.add_argument("--n_nets", default=10, type=int)
    parser.add_argument('--gp_rank', help='Rank of the task covar module', type=int, default=1)
    parser.add_argument('--kernel_type', help='Kernel for the GP', type=str, default='rbf')
    parser.add_argument('--n_test_episodes', help='Number of test episodes', type=int, default=10)
    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/gp/'
    else:
        experiment_dir = '/tmp/data/zhanghc/gp/'
    experiment_dir = experiment_dir + '1202/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_ver{}_thre{}_tau{}_d{}_n{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.version, args.target_threshold, args.tau, args.top_quantiles_to_drop_per_net, args.n_nets))

    file_name = experiment_dir + args.env + '/'
    if os.path.exists(file_name):
        shutil.rmtree(file_name)
    os.makedirs(file_name)


    env = gym.make(args.env)

    # Set seeds

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    offline_dataset = d4rl.qlearning_dataset(env)

    train_x = torch.from_numpy(offline_dataset['observations']).float().to(device)
    train_y = torch.from_numpy(offline_dataset['actions']).float().to(device)

    # Initialize likelihood and model
    gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=action_dim).to(device)

    gp_model = MultitaskGPModel(train_x, train_y, gp_likelihood, num_tasks=action_dim, rank=args.gp_rank,
                             ard_num_dims=state_dim, kernel_type=args.kernel_type).to(device)


    epochs = 1000

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likelihood, gp_model)
    gp_optimizer = torch.optim.Adam([
        {'params': gp_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    for epoch in range(epochs):
        gp_model.train()
        gp_likelihood.train()

        gp_optimizer.zero_grad()
        output = gp_model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        gp_optimizer.step()

        trainl, trains, avg_v = batch_assess(gp_model, gp_likelihood, train_x, train_y)

        print('Iter %d/%d - Loss: %.3f, Train mean log likelihood: %.3f, Train RMSE: %.3f' % (
            epoch, epochs, loss.item(), trainl, trains
        ))
        writer.add_scalar('loss/loss.item()', loss, epoch)
        writer.add_scalar('loss/trainl', trainl, epoch)
        writer.add_scalar('loss/trains', trains, epoch)

        if epoch % args.evaluation_interval == 0:
            gp_model.eval()
            gp_likelihood.eval()

            max_path_length = 1000

            start = datetime.datetime.now()

            ps = collect_new_paths(
                env,
                gp_model,
                gp_likelihood,
                max_path_length,
                max_path_length * args.n_test_episodes,
                discard_incomplete_paths=True,
            )

            finish = datetime.datetime.now()
            print("Profiling took: ", finish - start)

            eval_rew = np.mean([np.sum(p['rewards']) for p in ps])
            eval_std = np.std([np.sum(p['rewards']) for p in ps])
            print(f'Epoch {epoch}, Offline Return: {eval_rew}, Std: {eval_std}')
            writer.add_scalar('loss/eval_rew', eval_rew, epoch)
            writer.add_scalar('loss/eval_std', eval_std, epoch)

            torch.save(gp_model.state_dict(), f'{file_name}/gp_{args.kernel_type}_{epoch}.pt')





