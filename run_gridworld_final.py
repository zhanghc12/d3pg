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

alg = 'darl'
alg = 'mc_dropout'
alg = 'vae'
#alg = 'mopo'

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.reshape(-1)
fontsize = 44

if True:
    #plt.subplot(2, 2, 4)

    feature_nn, tree = darl.build_tree(post_observations, actions)
    with torch.no_grad():
        confidence = darl.get_uncertainty(X_grid[:,:1], X_grid[:,1:], feature_nn, tree)

    z = confidence.reshape(xx.shape)
    z = 1 - z.clip(0,0.2)

    # plt.figure()
    axes[3].contourf(x_lin, y_lin, z, 4, cmap='cividis')

    data = np.concatenate([post_observations, actions], axis=1)
    np.random.shuffle(data)
    axes[3].scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')  # marker='x')# , s=12)

    axes[3].set_title('Our method', fontsize=fontsize)

    # plt.show()

if True:
    # now we start to train the mc dropout or mopo
    #plt.subplot(2, 2, 3)

    z = 1
    qf1, qf2, vae_policy = mc_dropout.build_network(1, 1, '/Users/peixiaoqi/icml2022/gridworld', version=33)
    with torch.no_grad():
        confidence = mc_dropout.get_uncertainty(X_grid[:,:1], X_grid[:,1:], qf1, qf2, vae_policy)
    z = confidence.reshape(xx.shape)

    z = 1 - z
    #z = np.zeros_like(z)

    #plt.figure()#facecolor='#F1D511')
    #fig = plt.gcf()
    # fig.set_facecolor('#F1D511')
    axes[0].contourf(x_lin, y_lin, z, 4, cmap='cividis')

    data = np.concatenate([post_observations, actions], axis=1)
    np.random.shuffle(data)
    axes[0].scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')  # marker='x')# , s=12)
    axes[0].set_title('MC Dropout', fontsize=fontsize)

    # plt.show()

if True:
    # now we start to train the mc dropout or mopo
    #plt.subplot(2, 2, 2)

    z = 1
    qf1, qf2, vae_policy = mc_dropout.build_network(1, 1, '/Users/peixiaoqi/icml2022/gridworld', version=9)
    with torch.no_grad():
        confidence = mc_dropout.get_uncertainty(X_grid[:,:1], X_grid[:,1:], qf1, qf2, vae_policy, use_vae=True)
    z = confidence.reshape(xx.shape)
    z = - z
    #z = z


    #plt.figure()
    axes[2].contourf(x_lin, y_lin, z, 4, cmap='cividis')

    data = np.concatenate([post_observations, actions], axis=1)
    np.random.shuffle(data)
    axes[2].scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')#marker='x')# , s=12)
    axes[2].set_title('VAE', fontsize=fontsize)

    #plt.show()

if True:
    #plt.subplot(2, 2, 1)

    # now we start to train the mc dropout or mopo
    z = 1
    ensemble_model = mopo.build_model('/Users/peixiaoqi/icml2022/gridworld_', version=500)
    with torch.no_grad():
        confidence = mopo.get_uncertainty(ensemble_model, X_grid[:,:1], X_grid[:,1:])
    z = confidence.reshape(xx.shape)
    z = - z
    #z = z


    # plt.figure()
    axes[1].contourf(x_lin, y_lin, z, 10, cmap='cividis')

    data = np.concatenate([post_observations, actions], axis=1)
    np.random.shuffle(data)
    axes[1].scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')#marker='x')# , s=12)
    axes[1].set_title('Model Ensemble', fontsize=fontsize)

plt.show()

