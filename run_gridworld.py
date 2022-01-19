import env_gridworld
import gym
import numpy as np
from duelingpg.utils import ReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from uncertainty_demo.darl import build_tree, get_uncertainty
import os

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

save = True
load = not save
dirname = ''

"""
if torch.cuda.is_available():
    dirname = '/data/zhanghc/maze/maze.npy'
else:
    dirname = './tmp/data/zhanghc/maze/maze.npy'
"""

dirname = '/tmp/data/zhanghc/maze/maze.npy'
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

if alg == 'darl':
    feature_nn, tree = build_tree(post_observations, actions)
    with torch.no_grad():
        confidence = get_uncertainty(X_grid[:,:1], X_grid[:,1:], feature_nn, tree)

    z = confidence.reshape(xx.shape)
    z = 1 - z.clip(0,0.2)

if alg == 'mc_dropout':
    # now we start to train the mc dropout or mopo
    z = 1

plt.figure()
plt.contourf(x_lin, y_lin, z, 4, cmap='cividis')

data = np.concatenate([post_observations, actions], axis=1)
np.random.shuffle(data)
plt.scatter(data[:5000, 0], data[:5000, 1], marker='.', s=20, c='#EA7F45')#marker='x')# , s=12)

plt.show()

