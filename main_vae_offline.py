import argparse
import time
import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import duelingpg.utils as utils
import d4rl
import d4rl.gym_mujoco
from mc_dropout.drop import FlattenDropout_Mlp, VAEPolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    parser.add_argument('--test', type=int, default=0, metavar='A',
                        help='hyper or model_type')
    parser.add_argument('--version', type=int, default=0, metavar='A',
                        help='hyper or model_type')
    parser.add_argument('--seed', type=int, default=0, metavar='A',
                        help='hyper or model_type')
    return parser.parse_args()

def predict_uncertainty(vae, state_batch, action_batch):
    state_batch = torch.FloatTensor(state_batch).to(device=device)
    action_batch = torch.FloatTensor(action_batch).to(device=device)

    sampled_actions, raw_sampled_actions = vae.decode_multiple(state_batch, num_decode=100, device=device)
    distance = (raw_sampled_actions - action_batch.unsqueeze(1).repeat(1, 100, 1)) ** 2
    weight = torch.mean(torch.mean(distance, dim=1), dim=1).detach().cpu().numpy()
    # print(weight.shape)

    return weight

def test_uncertainty(memory, vae_policy, batch_size=2560):
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

        iid_distance = predict_uncertainty(vae_policy, state_batch, action_batch)
        ood_distance1 = predict_uncertainty(vae_policy, state_batch, ood_action_batch1)
        ood_distance2 = predict_uncertainty(vae_policy, state_batch, ood_action_batch2)
        ood_distance3 = predict_uncertainty(vae_policy, state_batch, ood_action_batch3)
        ood_distance4 = predict_uncertainty(vae_policy, state_batch, ood_action_batch4)

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
    obs_dim = state_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    offline_dataset = d4rl.qlearning_dataset(env)
    obs_mean, obs_std = load_hdf5(offline_dataset, replay_buffer)

    # Evaluate untrained policy
    if torch.cuda.is_available():
        log_dir = '/data/zhanghc/uncertainty/uwac/vae/'
        dirname = log_dir
    else:
        log_dir = '/tmp/data/zhanghc/uncertainty/uwac/vae'
        dirname = log_dir

    M = 256
    variant = {
        'drop_rate': 0.1,
        'spectral_norm': True
    }

    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[750, 750],
        latent_dim=action_dim * 2,
    ).to(device)

    vae_dirname = dirname + args.env_name + '/model_' + str(args.version)

    vae_policy.load_state_dict(torch.load(vae_dirname))


    test_uncertainty(replay_buffer, vae_policy)


if __name__ == '__main__':
    main()
