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

fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
axes = axes.reshape(-1)
fontsize = 44

if False:
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


    mean = [-1.7311e-02, -1.6706e-02, -1.6101e-02, -1.5495e-02, -1.4887e-02,
        -1.4277e-02, -1.3666e-02, -1.3048e-02, -1.2426e-02, -1.1802e-02,
        -1.1177e-02, -1.0540e-02, -9.8929e-03, -9.2452e-03, -8.5919e-03,
        -7.9342e-03, -7.2792e-03, -6.6250e-03, -5.9714e-03, -5.3188e-03,
        -4.6675e-03, -4.0176e-03, -3.3805e-03, -2.7789e-03, -2.5550e-03,
        -2.4988e-03, -2.4876e-03, -2.4805e-03, -2.4862e-03, -2.4919e-03,
        -2.4976e-03, -2.5067e-03, -2.5197e-03, -2.5326e-03, -2.5456e-03,
        -2.5586e-03, -2.5716e-03, -2.5845e-03, -2.5975e-03, -2.6105e-03,
        -2.6234e-03, -2.6364e-03, -2.6494e-03, -2.6631e-03, -2.6737e-03,
        -2.6708e-03, -2.6680e-03, -2.6651e-03, -2.6623e-03, -2.6594e-03,
        -2.6566e-03, -2.6539e-03, -2.6512e-03, -2.6484e-03, -2.6457e-03,
        -2.6430e-03, -2.6403e-03, -2.6376e-03, -2.6345e-03, -2.6306e-03,
        -2.6267e-03, -2.6207e-03, -2.6130e-03, -2.6055e-03, -2.5980e-03,
        -2.5906e-03, -2.5825e-03, -2.5708e-03, -2.5197e-03, -2.4387e-03,
        -2.2999e-03, -2.0310e-03, -1.7035e-03, -1.3113e-03, -8.8770e-04,
        -4.2709e-04,  4.5003e-05,  5.2624e-04,  1.0132e-03,  1.4988e-03,
         1.9901e-03,  2.4919e-03,  2.9881e-03,  3.4813e-03,  3.9710e-03,
         4.4475e-03,  4.8999e-03,  5.3216e-03,  5.7391e-03,  6.1508e-03,
         6.5607e-03,  6.9583e-03,  7.3531e-03,  7.7463e-03,  8.1305e-03,
         8.5067e-03,  8.8779e-03,  9.2472e-03,  9.6152e-03,  9.9702e-03]

    std = [1.0023, 1.0023, 1.0022, 1.0021, 1.0021, 1.0020, 1.0020, 1.0020, 1.0019,
        1.0019, 1.0018, 1.0018, 1.0018, 1.0018, 1.0017, 1.0017, 1.0017, 1.0017,
        1.0017, 1.0017, 1.0017, 1.0017, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016,
        1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016,
        1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016,
        1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016,
        1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016,
        1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016, 1.0016,
        1.0016, 1.0016, 1.0016, 1.0017, 1.0017, 1.0017, 1.0017, 1.0017, 1.0017,
        1.0017, 1.0017, 1.0017, 1.0018, 1.0018, 1.0018, 1.0018, 1.0019, 1.0019,
        1.0020, 1.0020, 1.0021, 1.0021, 1.0022, 1.0022, 1.0023, 1.0023, 1.0024,
        1.0025]



    mean = mean * 100
    std = std * 100

    z = np.zeros([10000, 1])
    for i in range(10000):
        z[i] = np.abs((X_grid[i, 1] - mean[i]) / std[i])

    z = z.reshape(xx.shape)

    z = 1 - z.clip(0, 1)
    #z = np.zeros_like(z)

    #plt.figure()#facecolor='#F1D511')
    #fig = plt.gcf()
    # fig.set_facecolor('#F1D511')
    axes[0].contourf(x_lin, y_lin, z, 4, cmap='cividis')

    data = np.concatenate([post_observations, actions], axis=1)
    np.random.shuffle(data)
    axes[0].scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')#marker='x')# , s=12)
    axes[0].set_title('DUE', fontsize=fontsize)


    # plt.show()

if True:
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
    axes[1].set_title('MOPO-style Ensemble', fontsize=fontsize)


if True:
    #plt.subplot(2, 2, 1)

    # now we start to train the mc dropout or mopo
    z = 1
    ensemble_model = mopo.build_model('/Users/peixiaoqi/icml2022/gridworld_', version=500)
    with torch.no_grad():
        confidence = mopo.get_uncertainty(ensemble_model, X_grid[:,:1], X_grid[:,1:], version=1)
    z = confidence.reshape(xx.shape)
    # z = - z
    z = 1 - z.clip(0,0.001)


    # plt.figure()
    axes[2].contourf(x_lin, y_lin, z, 4, cmap='cividis')

    data = np.concatenate([post_observations, actions], axis=1)
    np.random.shuffle(data)
    axes[2].scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')#marker='x')# , s=12)
    axes[2].set_title('Standard Ensemble', fontsize=fontsize)

plt.show()

