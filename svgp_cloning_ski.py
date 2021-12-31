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
from tqc import sn
import math
import torch.nn as nn
from tqc.spectral_normalization import spectral_norm
import torch.nn.functional as F
import torch
from torch.optim import SGD, Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, action_dim, num_feature):
        super(FeatureExtractor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, num_feature)

    def forward(self, data):
        q = F.relu(self.l1(data))
        q = F.relu(self.l2(q))
        #return spectral_norm(F.relu(self.l3(q)), norm_bound=0.95, n_power_iterations=1) # todo: if relu or not
        return self.l3(q)

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        # features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

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
    parser.add_argument('--evaluation_interval', help='Evaluation interval', type=int, default=10)
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
        train_len = int(len(offline_dataset['observations']) * 1)
        train_obs = torch.from_numpy(offline_dataset['observations'][:train_len]).float().to(device)
        train_act = torch.from_numpy(offline_dataset['actions'][:train_len]).float().to(device)
        train_x = torch.cat([train_obs, train_act], dim=1)

        train_y = torch.from_numpy(offline_dataset['rewards'][:train_len]).float().to(device)
        train_y = train_y.squeeze().unsqueeze(1)
        train_next_obs = torch.from_numpy(offline_dataset['next_observations'][:train_len]).float().to(device)
        train_y = torch.cat([train_next_obs - train_obs, train_y], dim=1)

        #test_obs = torch.from_numpy(offline_dataset['observations'][train_len:]).float().to(device)
        #test_act = torch.from_numpy(offline_dataset['actions'][train_len:]).float().to(device)

        #test_x = torch.cat([test_obs, test_act], dim=1)

        #test_y = torch.from_numpy(offline_dataset['rewards'][train_len:]).float().to(device)
        #test_y = test_y.squeeze().unsqueeze(1)
        # test_next_obs = torch.from_numpy(offline_dataset['next_observations'][train_len:]).float().to(device)
        # test_y = torch.cat([test_next_obs - test_obs, test_y], dim=1)


    else:
        train_len = int(len(offline_dataset['observations']) * 1)
        train_obs = torch.from_numpy(offline_dataset['observations'][:train_len]).float().to(device)
        train_act = torch.from_numpy(offline_dataset['actions'][:train_len]).float().to(device)
        train_x = torch.cat([train_obs, train_act], dim=1)
        train_x = train_x[:1000]
        train_y = torch.from_numpy(offline_dataset['rewards'][:train_len]).float().to(device)
        train_y = train_y.squeeze().unsqueeze(1)
        train_next_obs = torch.from_numpy(offline_dataset['next_observations'][:train_len]).float().to(device)
        train_y = torch.cat([train_next_obs - train_obs, train_y], dim=1)
        train_y = train_y[:1000]

        #train_x = torch.from_numpy(offline_dataset['observations'][:1000]).float().to(device)
        #train_y = torch.from_numpy(offline_dataset['actions'][:1000]).float().to(device)

        #test_x = torch.from_numpy(offline_dataset['observations'][:1000]).float().to(device)
        #test_y = torch.from_numpy(offline_dataset['actions'][:1000]).float().to(device)
    # Initialize likelihood and model

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


    #test_dataset = TensorDataset(test_x, test_y)
    #test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    feature_extractor = FeatureExtractor(state_dim=state_dim, action_dim=action_dim, num_feature=train_y.shape[1])

    model = DKLModel(feature_extractor=feature_extractor, num_dim=train_y.shape[1]).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(device)

    if torch.cuda.is_available():
        model = model.to(device)

    model.train()
    likelihood.train()

    lr = 0.1
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=train_y.size(0)).to(device)

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

        '''
        model.eval()
        likelihood.eval()
        
        means = torch.tensor([0.])
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = model(x_batch)
                predictions = likelihood(preds)
                mean = predictions.mean

                lower, upper = predictions.confidence_region()
        # means = means[1:]
        '''


        # trainl, trains, avg_v = batch_assess(model, likelihood, train_x, train_y)

        print('Iter %d/%d - Loss: %.3f, Train mean log likelihood: %.3f, Train RMSE: %.3f' % (
            epoch, epochs, loss.item(), loss.item(), loss.item()
        ))
        writer.add_scalar('loss/loss.item()', loss, epoch)
        #writer.add_scalar('loss/trainl', trainl, epoch)
        #writer.add_scalar('loss/trains', trains, epoch)
        if epoch % args.evaluation_interval == 0:
            torch.save(model.state_dict(), f'{file_name}/gp_{args.kernel_type}_{epoch}.pt')


