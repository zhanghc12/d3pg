from offline_kd_test_v3 import FeatureExtractorV4
import torch
from sklearn.neighbors import KDTree
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tree(state, action, dim=9): # note, state and action is numpy
    state_dim = state.shape[1]
    action_dim = action.shape[1]
    feature_nn = FeatureExtractorV4(state_dim, action_dim, 256, 10).to(device)

    size = state.shape[0]
    i = 0
    batch_size = 2560
    phi_list = []

    while i + batch_size < size:
        index = np.arange(i, i + batch_size)
        state_batch = state[index]
        action_batch = action[index]

        phi = feature_nn(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(action_batch).to(device)).detach().cpu().numpy()
        phi_list.extend(phi)
        i += batch_size

    tree = KDTree(np.array(phi_list), leaf_size=40)
    # after build tree
    return feature_nn, tree

def get_uncertainty(state, action, feature_nn, tree):
    distances = []

    state_batch = state
    action_batch = action

    phi = feature_nn(torch.FloatTensor(state_batch).to(device),
                     torch.FloatTensor(action_batch).to(device)).detach().cpu().numpy()
    distance = tree.query(phi, k=1)[0][:, :1]
    # distance = np.mean(distance, axis=1, keepdims=True)

    return distance