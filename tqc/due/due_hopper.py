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
import seaborn as sns

import env_gridworld
import gym
import numpy as np
from duelingpg.utils import ReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from uncertainty_demo import darl
import os
from uncertainty_demo import mc_dropout
from uncertainty_demo import mopo


seed = 4
torch.manual_seed(seed)
np.random.seed(seed)
sns.set()

env = gym.make('GridWorld-v0')

episode_nums = 100
max_episode_steps = 100
state = env.reset()

state_dim = 1
action_dim = 1
replay_buffer = ReplayBuffer(state_dim, action_dim)

# right action
for episode in range(episode_nums):
    step = 0
    done = False
    while step < max_episode_steps:
        action = np.random.choice([-1,1], size=1) + 0.05 * np.random.normal(0., 1., size=1) # np.array([1])
        next_state, reward, done, info = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done, done)
        state = next_state
        step += 1
        if done:
            state = env.reset()
            break
        # print(reward)

# left action
for episode in range(episode_nums):
    step = 0
    done = False
    while step < max_episode_steps:
        action = np.random.choice([-1,1], size=1) + 0.05 * np.random.normal(0., 1., size=1)
        next_state, reward, done, info = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done, done)
        state = next_state
        step += 1
        if done:
            state = env.reset()
            break

        # print(reward)

print('setup dataset')
# save the dataset
observations = replay_buffer.state[:replay_buffer.ptr, :]
mean_obs = np.mean(observations, axis=0)
std_obs = np.std(observations, axis=0)
actions = replay_buffer.action[:replay_buffer.ptr, :]
next_observations = replay_buffer.next_state[:replay_buffer.ptr, :]
not_done = replay_buffer.not_done[:replay_buffer.ptr, :]
reward = replay_buffer.reward[:replay_buffer.ptr, :]

save = False
load = not save
dirname = ''

"""
if torch.cuda.is_available():
    dirname = '/data/zhanghc/maze/maze.npy'
else:
    dirname = './tmp/data/zhanghc/maze/maze.npy'
"""

# dirname = '/tmp/data/zhanghc/maze/maze.npy'

if torch.cuda.is_available():
    dirname = './tqc/due/maze.npy'
else:
    dirname = '/Users/peixiaoqi/icml2022/maze.npy'

if save:
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))
    dataset = {'observations': observations, 'actions': actions, 'next_observations':next_observations, 'rewards':reward,
          'terminals': 1 - not_done}
    np.save(dirname, dataset)
    print('saved')
if load:
    dataset = np.load(dirname, allow_pickle=True)
    observations = dataset.item()['observations']
    actions = dataset.item()['actions']
    print('loaded')

post_observations = (observations - mean_obs) / (std_obs + 1e-5)

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

x_lin = np.linspace(-2.5, 2.5, 100)
y_lin = np.linspace(-2, 2, 100)
xx, yy = np.meshgrid(x_lin, y_lin)
X_grid = np.column_stack([xx.flatten(), yy.flatten()])

alg = 'due'

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.reshape(-1)
fontsize = 44


domain = 15

x, y = post_observations, actions
n_samples = len(x)



batch_size = 128

X_train, y_train = x, y[:, 0]
X_test, y_test = np.expand_dims(x_lin, 1), y_lin

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False)

steps = 10e3
epochs = steps // len(dl_train) + 1
print(f"Training with {n_samples} datapoints for {epochs} epochs")

# Change this boolean to False for SNGP
DUE = True

input_dim = post_observations.shape[1]
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
    if DUE:
        likelihood = likelihood.cuda()

lr = 1e-2

parameters = [
    {"params": model.parameters(), "lr": lr},
]
parameters.append({"params": likelihood.parameters(), "lr": lr})
optimizer = torch.optim.Adam(parameters)
pbar = ProgressBar()

def step(engine, batch):
    model.train()
    if DUE:
        likelihood.train()

    optimizer.zero_grad()

    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    y_pred = model(x)

    if not DUE:
        y_pred.squeeze_()

    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_step(engine, batch):
    model.eval()
    if DUE:
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

trainer.run(dl_train, max_epochs=epochs)

model.eval()
likelihood.eval()

# x_lin = np.linspace(-domain, domain, 100)

with torch.no_grad():#, gpytorch.settings.num_likelihood_samples(64):
    # xx = torch.tensor(x_lin[..., None]).float()

    xx = torch.tensor(X_test).float()

    if torch.cuda.is_available():
        xx = xx.cuda()
    pred = model(xx)

    ol = likelihood(pred)
    output = ol.mean.cpu()
    output_std = ol.stddev.cpu()
    print(done)
    print(output)
    print(output_std)

