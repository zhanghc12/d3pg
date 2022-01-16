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
    parser.add_argument('--gp_rank', help='Rank of the task covar module', type=int, default=1)
    parser.add_argument('--kernel_type', help='Kernel for the GP', type=str, default='rbf')
    parser.add_argument('--n_test_episodes', help='Number of test episodes', type=int, default=10)
    parser.add_argument('--num_points', help='evaluation period', type=int, default=10)
    parser.add_argument('--evaluation_interval', help='evaluation period', type=int, default=10)

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/gp/'
    else:
        experiment_dir = '/tmp/data/zhanghc/gp/'
    experiment_dir = experiment_dir + '0116/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}_n{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed, args.num_points))

    file_name = experiment_dir + args.env + '/' + str(args.num_points)
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
    train_len = int(len(offline_dataset['observations']) * 1)
    train_obs = torch.from_numpy(offline_dataset['observations'][:train_len]).float().to(device)
    train_act = torch.from_numpy(offline_dataset['actions'][:train_len]).float().to(device)
    train_obs = (train_obs - torch.mean(train_obs, dim=0)) / (torch.std(train_obs, dim=0) + 1e-5)
    train_x = torch.cat([train_obs, train_act], dim=1)
    # train_x = (train_x - torch.mean(train_x, dim=0)) / (torch.std(train_x, dim=0) + 1e-5)

    train_y = torch.from_numpy(offline_dataset['rewards'][:train_len]).float().to(device)
    train_y = train_y.squeeze().unsqueeze(1)
    train_next_obs = torch.from_numpy(offline_dataset['next_observations'][:train_len]).float().to(device)
    train_y = torch.cat([train_next_obs - train_obs, train_y], dim=1)

    if not torch.cuda.is_available():
        train_x = train_x[:1000]
        train_y = train_y[:1000]

    # Initialize likelihood and model
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    inducing_points = train_x[:args.num_points, :] # todo

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

        print('Iter %d/%d - Loss: %.3f, Train mean log likelihood: %.3f, Train RMSE: %.3f' % (
            epoch, epochs, loss.item(), loss.item(), loss.item()
        ))
        writer.add_scalar('loss/loss.item()', loss, epoch)

        if epoch % args.evaluation_interval == 0:
            torch.save(model.state_dict(), f'{file_name}/gp_{args.kernel_type}_{epoch}.pt')
            torch.save(likelihood.state_dict(), f'{file_name}/lk_{args.kernel_type}_{epoch}.pt')


