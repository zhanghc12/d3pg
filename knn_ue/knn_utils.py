import torch
import numpy as np

class RMS:
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    # used for a batch, but how to determine the uncertainty ,given the , to get inducing point? to fully denote the matrix
    # one dimension
    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward


def test_tree(memory, kd_tree, k=3, batch_size=256):
    i = 0
    iid_list = []
    ood_list1 = []
    ood_list2 = []

    size = 0
    if torch.cuda.is_available():
        size = memory.size
    else:
        size = 1000
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_data = np.concatenate([state_batch, action_batch], axis=1)
        iid_distance = kd_tree.query(iid_data, k=k)[0]

        ood_action_batch1 = action_batch + 0.1 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch1 = np.clip(ood_action_batch1, -1, 1)
        ood_data1 = np.concatenate([state_batch, ood_action_batch1], axis=1)

        ood_action_batch2 = action_batch + 1 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch2 = np.clip(ood_action_batch2, -1, 1)
        ood_data2 = np.concatenate([state_batch, ood_action_batch2], axis=1)

        ood_distance1 = kd_tree.query(ood_data1, k=k)[0]
        ood_distance2 = kd_tree.query(ood_data2, k=k)[0]

        iid_distance = np.mean(iid_distance, axis=1, keepdims=False)
        ood_distance1 = np.mean(ood_distance1, axis=1, keepdims=False)
        ood_distance2 = np.mean(ood_distance2, axis=1, keepdims=False)

        iid_list.extend(iid_distance)
        ood_list1.extend(ood_distance1)
        ood_list2.extend(ood_distance2)

        i += batch_size

        print("iid: {:4f}, ood1: {:4f}, ood2: {:4f}".format(np.mean(iid_distance), np.mean(ood_distance1), np.mean(ood_distance1)))
    return np.mean(iid_list)


