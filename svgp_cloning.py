import d4rl
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks=4):
        # Let's use a different set of inducing points for each task
        # inducing_points = torch.rand(num_tasks, 16, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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

    if torch.cuda.is_available():
        train_len = int(len(offline_dataset['observations']) * 0.9)
        train_x = torch.from_numpy(offline_dataset['observations'][:train_len]).float().to(device)
        train_y = torch.from_numpy(offline_dataset['actions'][:train_len]).float().to(device)

        test_x = torch.from_numpy(offline_dataset['observations'][train_len:]).float().to(device)
        test_y = torch.from_numpy(offline_dataset['actions'][train_len:]).float().to(device)
    else:
        train_x = torch.from_numpy(offline_dataset['observations'][:1000]).float().to(device)
        train_y = torch.from_numpy(offline_dataset['actions'][:1000]).float().to(device)

        test_x = torch.from_numpy(offline_dataset['observations'][:1000]).float().to(device)
        test_y = torch.from_numpy(offline_dataset['actions'][:1000]).float().to(device)
    # Initialize likelihood and model

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    inducing_points = train_x[:5000, :] # todo

    inducing_points = inducing_points.unsqueeze(0).repeat(train_y.shape[1], 1, 1)
    model = IndependentMultitaskGPModel(inducing_points=inducing_points, num_tasks=train_y.shape[1])
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(device)

    if torch.cuda.is_available():
        model = model.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0)).to(device)

    epochs = 1000

    for epoch in range(epochs):
        model.train()
        likelihood.train()

        for x_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            # print(output.device)
            # print(y_batch.device)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = model(x_batch)
                means = torch.cat([means, preds.mean.cpu()])
        means = means[1:]


        # trainl, trains, avg_v = batch_assess(model, likelihood, train_x, train_y)

        print('Iter %d/%d - Loss: %.3f, Train mean log likelihood: %.3f, Train RMSE: %.3f' % (
            epoch, epochs, loss.item(), loss.item(), loss.item()
        ))
        writer.add_scalar('loss/loss.item()', loss, epoch)
        #writer.add_scalar('loss/trainl', trainl, epoch)
        #writer.add_scalar('loss/trains', trains, epoch)

        torch.save(model.state_dict(), f'{file_name}/gp_{args.kernel_type}_{epoch}.pt')

        '''
        if epoch % args.evaluation_interval == 0:
            model.eval()
            likelihood.eval()

            max_path_length = 1000

            start = datetime.datetime.now()

            ps = collect_new_paths(
                env,
                model,
                likelihood,
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
            
            torch.save(model.state_dict(), f'{file_name}/gp_{args.kernel_type}_{epoch}.pt')

            '''

