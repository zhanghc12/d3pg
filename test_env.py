import gym
import d4rl.gym_mujoco
import numpy as np
import duelingpg.utils as utils


def normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return (data - mean) / (std + 1e-5)


env1 = gym.make('hopper-expert-v0')
env2 = gym.make('halfcheetah-expert-v0')
env3 = gym.make('ant-expert-v0')

for env in [env1, env2, env3]:
    print('env', env.observation_space.shape[0] + env.action_space.shape[0])


def load_hdf5(dataset, replay_buffer):
    #obs_mean = np.mean(np.abs(dataset['next_observations'] - dataset['observations']), axis=0, keepdims=True)
    #obs_std = np.std(np.abs(dataset['next_observations'] - dataset['observations']), axis=0, keepdims=True)

    obs_mean = np.mean(dataset['observations'], axis=0, keepdims=True)
    obs_std = np.std(dataset['observations'], axis=0, keepdims=True)


    replay_buffer.state = normalize(dataset['observations'])
    replay_buffer.action = dataset['actions']
    replay_buffer.next_state = normalize(dataset['next_observations'])
    replay_buffer.reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer.not_done = 1 - np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer.next_action = np.concatenate([dataset['actions'][1:],dataset['actions'][-1:]], axis=0)
    replay_buffer.forward_label = normalize(np.concatenate([dataset['next_observations'] - dataset['observations'], np.expand_dims(np.squeeze(dataset['rewards']), 1)], axis=1))

    rel_obs = dataset['next_observations'] - dataset['observations']
    print(np.std(rel_obs, axis=0, keepdims=True))
    print(np.mean(np.std(rel_obs, axis=0, keepdims=True)))

    rew_mean = np.mean(replay_buffer.reward, axis=0, keepdims=True)
    rew_std = np.std(replay_buffer.reward, axis=0, keepdims=True)

    replay_buffer.size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer.not_done.sum())
    return obs_mean, obs_std, rew_mean, rew_std

env = gym.make('hopper-medium-replay-v0')
#env = gym.make('halfcheetah-random-v0')

#env = gym.make('hopper-expert-v0')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
offline_dataset = d4rl.qlearning_dataset(env)
obs_mean, obs_std, rew_mean, rew_std = load_hdf5(offline_dataset, replay_buffer)
#for std in obs_std[0]:
#    print('{:.4f},'.format(std), end="")
#print('')

#print(obs_mean, obs_std, rew_mean, rew_std)

# hopper 0.07
# medium
# medium-replay be small
# medium-expert, expert must be smaller

# halfcheetah 0.005 * 23 = 0.115

# change another thing

# -> 0.1 not good -> 0.05, be 20
# [0.005, 0.07], must be 0.02, 0.05
# clip must be 50, 20 enough