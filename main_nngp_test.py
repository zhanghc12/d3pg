import d4rl
import d4rl.gym_mujoco
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
import duelingpg.utils as utils


'''
todo 1: fix argument, todo 2: normalize the observation
todo 3: check gpgnn
different version: svgp has done
then svgp +nn 
then ski ? we have given up
'''


def normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data - mean) / (std + 1e-5)

def load_hdf5(dataset, replay_buffer):
    obs_mean = np.mean(dataset['observations'], axis=0, keepdims=True)
    obs_std = np.std(dataset['observations'], axis=0, keepdims=True)
    replay_buffer.state = normalize(dataset['observations'])
    # replay_buffer.action = normalize(dataset['actions'])
    replay_buffer.action = dataset['actions']

    print(np.std(dataset['actions'], axis=0, keepdims=True))

    replay_buffer.action = normalize(dataset['actions'])

    replay_buffer.next_state = normalize(dataset['next_observations'])
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer.next_action = np.concatenate([dataset['actions'][1:],dataset['actions'][-1:]], axis=0)
    replay_buffer.forward_label = normalize(np.concatenate([dataset['next_observations'] - dataset['observations'], np.expand_dims(np.squeeze(dataset['rewards']), 1)], axis=1))

    replay_buffer.size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer.not_done.sum())
    return obs_mean, obs_std



def predict_uncertainty(model, likelihood, state_batch, action_batch):
    state_batch = torch.FloatTensor(state_batch).to(device=device)
    action_batch = torch.FloatTensor(action_batch).to(device=device)

    preds = model(torch.cat([state_batch, action_batch], dim=1))
    predictions = likelihood(preds)
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    weight = (upper - lower).squeeze().detach().cpu().numpy()
    return weight

def test_uncertainty(memory, model, likelihood, batch_size=2560):
    i = 0
    iid_list = []
    ood_list1 = []
    ood_list2 = []
    ood_list3 = []
    ood_list4 = []

    size = memory.size
    # size = 20000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(memory.state_dim + memory.action_dim)
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        ood_action_batch1 = action_batch + 0.01 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch1 = np.clip(ood_action_batch1, -1, 1)
        ood_action_batch2 = action_batch + 0.1 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch2 = np.clip(ood_action_batch2, -1, 1)
        ood_action_batch3 = action_batch + 0.3 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch3 = np.clip(ood_action_batch3, -1, 1)
        ood_action_batch4 = action_batch + 1.0 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch4 = np.clip(ood_action_batch4, -1, 1)

        iid_distance = predict_uncertainty(model, likelihood, state_batch, action_batch)
        ood_distance1 = predict_uncertainty(model, likelihood, state_batch, ood_action_batch1)
        ood_distance2 = predict_uncertainty(model, likelihood, state_batch, ood_action_batch2)
        ood_distance3 = predict_uncertainty(model, likelihood, state_batch, ood_action_batch3)
        ood_distance4 = predict_uncertainty(model, likelihood, state_batch, ood_action_batch4)

        iid_list.extend(iid_distance)
        ood_list1.extend(ood_distance1)
        ood_list2.extend(ood_distance2)
        ood_list3.extend(ood_distance3)
        ood_list4.extend(ood_distance4)

        i += batch_size

        print("step:{}, iid: {:4f}, ood1: {:4f}, ood2: {:4f}, ood3: {:4f}, ood4: {:4f}".format(i, np.mean(iid_distance), np.mean(ood_distance1), np.mean(ood_distance2), np.mean(ood_distance3), np.mean(ood_distance4)))


    iid_list = np.sort(iid_list)
    ood_list1 = np.sort(ood_list1)
    ood_list2 = np.sort(ood_list2)
    ood_list3 = np.sort(ood_list3)
    ood_list4 = np.sort(ood_list4)

    iid_list = iid_list #/ (memory.state_dim + memory.action_dim)
    ood_list1 = ood_list1 #/ (memory.state_dim + memory.action_dim)
    ood_list2 = ood_list2 #/ (memory.state_dim + memory.action_dim)
    ood_list3 = ood_list3 #/ (memory.state_dim + memory.action_dim)
    ood_list4 = ood_list4 #/ (memory.state_dim + memory.action_dim)


    print("0%: ", iid_list[0])
    print("0.1%: ", iid_list[np.int32(len(iid_list)*0.001)], ood_list1[np.int32(len(ood_list1)*0.001)], ood_list2[np.int32(len(ood_list2)*0.001)], ood_list3[np.int32(len(ood_list3)*0.001)], ood_list4[np.int32(len(ood_list4)*0.001)])
    print("1%: ", iid_list[np.int32(len(iid_list)*0.01)], ood_list1[np.int32(len(ood_list1)*0.01)], ood_list2[np.int32(len(ood_list2)*0.01)], ood_list3[np.int32(len(ood_list3)*0.01)], ood_list4[np.int32(len(ood_list4)*0.01)])
    print("10%: ", iid_list[np.int32(len(iid_list)*0.1)], ood_list1[np.int32(len(ood_list1)*0.1)], ood_list2[np.int32(len(ood_list2)*0.1)], ood_list3[np.int32(len(ood_list3)*0.1)], ood_list4[np.int32(len(ood_list4)*0.1)])
    print("20%: ", iid_list[np.int32(len(iid_list)*0.2)], ood_list1[np.int32(len(ood_list1)*0.2)], ood_list2[np.int32(len(ood_list2)*0.2)], ood_list3[np.int32(len(ood_list3)*0.2)], ood_list4[np.int32(len(ood_list4)*0.2)])
    print("50%: ", iid_list[np.int32(len(iid_list)*0.5)], ood_list1[np.int32(len(ood_list1)*0.5)], ood_list2[np.int32(len(ood_list2)*0.5)], ood_list3[np.int32(len(ood_list3)*0.5)], ood_list4[np.int32(len(ood_list4)*0.5)])
    print("99%: ", iid_list[np.int32(len(iid_list)*0.99)], ood_list1[np.int32(len(ood_list1)*0.99)], ood_list2[np.int32(len(ood_list2)*0.99)], ood_list3[np.int32(len(ood_list3)*0.99)], ood_list4[np.int32(len(ood_list4)*0.99)])
    print("100%: ", iid_list[-1], ood_list1[-1], ood_list2[-1], ood_list3[-1], ood_list4[-1])

    return iid_list


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
    parser.add_argument('--num_points', help='evaluation period', type=int, default=500)

    args = parser.parse_args()

    if torch.cuda.is_available():
        experiment_dir = '/data/zhanghc/gp/' + str(args.num_points) + '/'
    else:
        experiment_dir = '/tmp/data/zhanghc/gp/' + str(args.num_points) + '/'
    experiment_dir = experiment_dir + '0107/'
    writer = SummaryWriter(
        experiment_dir + '{}_{}_{}_s{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.policy, args.env, args.seed))

    file_name = experiment_dir + args.env + '/'

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
    feature_extractor = FeatureExtractor(state_dim=state_dim, action_dim=action_dim, num_feature=128).to(device)
    if not torch.cuda.is_available():
        inducing_points = train_x[:args.num_points, :]  # todo
    else:
        inducing_points = train_x[:args.num_points, :] # todo

    inducing_points = feature_extractor(inducing_points).to(device)  # initialize the inducing points
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

    model.load_state_dict(torch.load(f'{file_name}/gp_{args.kernel_type}_990.pt'))

    epochs = 1000

    model.eval()
    likelihood.eval()

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    offline_dataset = d4rl.qlearning_dataset(env)
    obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

    test_uncertainty(replay_buffer, model, likelihood)
