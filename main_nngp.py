import d4rl
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import gym
import argparse
import gpytorch
import os
import shutil
import torch
from torch.optim import SGD, Adam
from mygp.neural_gp import FeatureExtractor, NeuralGP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
todo 1: fix argument, todo 2: normalize the observation
todo 3: check gpgnn
different version: svgp has done
then svgp +nn 
then ski ? we have given up
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="nngp")  # Policy name (TD3, DDPG or OurDDPG, Dueling)
    parser.add_argument("--env", default="hopper-expert-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--gp_rank', help='Rank of the task covar module', type=int, default=1)
    parser.add_argument('--kernel_type', help='Kernel for the GP', type=str, default='rbf')
    parser.add_argument('--n_test_episodes', help='Number of test episodes', type=int, default=10)
    parser.add_argument('--enable_ngd', help='use ngd or not', type=int, default=1)
    parser.add_argument('--evaluation_interval', help='evaluation period', type=int, default=10)

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/gp/'
    else:
        experiment_dir = '/tmp/data/zhanghc/gp/'
    experiment_dir = experiment_dir + '0107/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed))

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
    train_len = int(len(offline_dataset['observations']) * 1)
    train_obs = torch.from_numpy(offline_dataset['observations'][:train_len]).float().to(device)
    train_act = torch.from_numpy(offline_dataset['actions'][:train_len]).float().to(device)
    train_x = torch.cat([train_obs, train_act], dim=1)
    train_x = (train_x - torch.mean(train_x, dim=0)) / (torch.std(train_x, dim=0) + 1e-5)

    train_y = torch.from_numpy(offline_dataset['rewards'][:train_len]).float().to(device)
    train_y = train_y.squeeze().unsqueeze(1)
    train_next_obs = torch.from_numpy(offline_dataset['next_observations'][:train_len]).float().to(device)
    train_y = torch.cat([train_next_obs - train_obs, train_y], dim=1)

    if not torch.cuda.is_available():
        train_x = train_x[:1000]
        train_y = train_y[:1000]
    test_x = train_x[:1000]
    test_y = train_y[:1000]

    # Initialize likelihood and model
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    # define the model
    feature_extractor = FeatureExtractor(state_dim=state_dim, action_dim=action_dim, num_feature=128)
    if not torch.cuda.is_available():
        inducing_points = train_x[:500, :]  # todo
    else:
        inducing_points = train_x[:5000, :] # todo

    inducing_points = feature_extractor(inducing_points)  # initialize the inducing points
    inducing_points = inducing_points.unsqueeze(0).repeat(train_y.shape[1], 1, 1)  # y_shape * ibatch * 128
    model = NeuralGP(feature_extractor, inducing_points, num_tasks=train_y.shape[1], enable_ngd=args.enable_ngd).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(device)

    model.train()
    likelihood.train()

    # define the training parameter
    variational_ngd_optimizer = gpytorch.optim.NGD(model.gp_layer.variational_parameters(), num_data=train_y.size(0),
                                                   lr=0.1)
    lr = 0.1
    if args.enable_ngd:
        optimizer = SGD([
            {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
            {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
            {'params': likelihood.parameters()},
        ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    else:
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
            variational_ngd_optimizer.zero_grad()
            output = model(x_batch)
            # print(output.device)
            # print(y_batch.device)
            loss = -mll(output, y_batch)
            loss.backward()
            variational_ngd_optimizer.step()
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
        # writer.add_scalar('loss/trainl', trainl, epoch)
        # writer.add_scalar('loss/trains', trains, epoch)
        if epoch % args.evaluation_interval == 0:
            torch.save(model.state_dict(), f'{file_name}/gp_{args.kernel_type}_{epoch}.pt')


