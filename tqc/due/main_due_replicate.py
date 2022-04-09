import argparse
import time
import gym
import torch
import numpy as np
import datetime
from mbpo_py.model_replicate_offline import EnsembleDynamicsModel
from mbpo_py.predict_env import PredictEnv
from mbpo_py.tf_models.constructor import construct_model
import duelingpg.utils as utils
import d4rl
import d4rl.gym_mujoco

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from tqc.due.dkl import DKL, GP, initial_values
from tqc.due.sngp import Laplace
from tqc.due.fc_resnet import FCResNet

import matplotlib.pyplot as plt

import gym
import numpy as np
from duelingpg.utils import ReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
import torch

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

    replay_buffer.next_state = normalize(dataset['next_observations'])
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer.next_action = np.concatenate([dataset['actions'][1:],dataset['actions'][-1:]], axis=0)
    replay_buffer.forward_label = normalize(np.concatenate([dataset['next_observations'] - dataset['observations'], np.expand_dims(np.squeeze(dataset['rewards']), 1)], axis=1))

    replay_buffer.size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer.not_done.sum())
    return obs_mean, obs_std


def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="hopper-expert-v0",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=25, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')

    parser.add_argument('--version', type=int, default=0, metavar='A',
                        help='hyper or model_type')

    parser.add_argument('--test', type=int, default=0, metavar='A',
                        help='hyper or model_type')

    return parser.parse_args()

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
    abnormal = [0, 0, 0, 0]
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

        d1 = np.mean(np.sqrt((ood_action_batch1 - action_batch) * (ood_action_batch1 - action_batch)), axis=-1)
        abnormal[0] = abnormal[0] + np.sum((d1 > 0.1).astype(float))

        d2 = np.mean(np.sqrt((ood_action_batch2 - action_batch) * (ood_action_batch2 - action_batch)), axis=-1)
        abnormal[1] = abnormal[1] + np.sum((d2 > 0.1).astype(float))

        d3 = np.mean(np.sqrt((ood_action_batch3 - action_batch) * (ood_action_batch3 - action_batch)), axis=-1)
        abnormal[2] = abnormal[2] + np.sum((d3 > 0.1).astype(float))

        d4 = np.mean(np.sqrt((ood_action_batch4 - action_batch) * (ood_action_batch4 - action_batch)), axis=-1)
        abnormal[3] = abnormal[3] + np.sum((d4 > 0.1).astype(float))

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

    print(abnormal)
    return iid_list


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Intial agent
    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    offline_dataset = d4rl.qlearning_dataset(env)
    obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

    batch_size = 256
    X_train, y_train = replay_buffer.state, replay_buffer.action
    y_train =  replay_buffer.action[:, 0]
    X_test = replay_buffer.state
    #X_test = X_test[:10000]

    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=True)

    y_test = y_train
    ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False)

    steps = 10e3
    epochs = steps // len(dl_train) + 1
    epochs = args.num_epoch
    print(f"Training with datapoints for {epochs} epochs")
    input_dim = X_train.shape[1]
    features = 128
    depth = 4
    num_outputs = 1  # regression with 1D output
    spectral_normalization = True
    coeff = 0.95
    n_power_iterations = 1
    dropout_rate = 0.01

    feature_extractor = FCResNet(
        input_dim=input_dim,
        features=features,
        depth=depth,
        spectral_normalization=spectral_normalization,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        dropout_rate=dropout_rate
    )

    n_inducing_points = 20
    kernel = "RBF"

    initial_inducing_points, initial_lengthscale = initial_values(
        ds_train, feature_extractor, n_inducing_points
    )

    gp = GP(
        num_outputs=num_outputs,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=kernel,
    )

    model = DKL(feature_extractor, gp)

    likelihood = GaussianLikelihood()
    elbo_fn = VariationalELBO(likelihood, model.gp, num_data=len(ds_train))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    lr = 1e-3

    parameters = [
        {"params": model.parameters(), "lr": lr},
    ]
    parameters.append({"params": likelihood.parameters(), "lr": lr})
    optimizer = torch.optim.Adam(parameters)
    pbar = ProgressBar()

    def step(engine, batch):
        model.train()
        likelihood.train()

        optimizer.zero_grad()

        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        likelihood.eval()

        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")
    pbar.attach(trainer)

    metric = Loss(lambda y_pred, y: - likelihood.expected_log_prob(y, y_pred).mean())
    metric.attach(evaluator, "loss")

    @trainer.on(Events.EPOCH_COMPLETED(every=int(epochs / 10) + 1))
    def log_results(trainer):
        evaluator.run(dl_test)
        print(f"Results - Epoch: {trainer.state.epoch} - "
              f"Test Likelihood: {evaluator.state.metrics['loss']:.2f} - "
              f"Loss: {trainer.state.metrics['loss']:.2f}")

    if args.test == 0:
        trainer.run(dl_train, max_epochs=epochs)
        torch.save(model.state_dict(), 'due' + args.env_name + '_m.pt')
        torch.save(likelihood.state_dict(), 'due' + args.env_name + '_l.pt')

    if args.test == 1:
        model.load_state_dict(torch.load('due' + args.env_name + '_m.pt'))
        likelihood.load_state_dict(torch.load('due' + args.env_name + '_l.pt'))

        model.eval()
        likelihood.eval()

    test_uncertainty(replay_buffer, model, likelihood)

def predict_uncertainty(model, likelihood, state, action):
    with torch.no_grad():  # , gpytorch.settings.num_likelihood_samples(64):
        # xx = torch.tensor(x_lin[..., None]).float()

        xx = torch.tensor(state).float()

        if torch.cuda.is_available():
            xx = xx.cuda()
        pred = model(xx)

        ol = likelihood(pred)
        output = ol.mean.cpu().numpy()
        output = np.squeeze(output)
        output_std = ol.stddev.cpu().numpy()
        output_std = np.squeeze(output_std)

        return np.abs((action[:, 0] - output) / (output_std + 1e-3))

if __name__ == '__main__':
    main()
