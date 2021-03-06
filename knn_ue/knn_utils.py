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


def test_tree(memory, kd_tree, k=3, batch_size=2560):
    i = 0
    iid_list = []
    ood_list1 = []
    ood_list2 = []

    size = 0
    if torch.cuda.is_available():
        size = memory.size
        size = 50000
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

        print("step:{}, iid: {:4f}, ood1: {:4f}, ood2: {:4f}".format(i, np.mean(iid_distance), np.mean(ood_distance1), np.mean(ood_distance2)))
    return iid_list


def test_tree_true(memory, kd_tree, k=3, batch_size=2560):
    i = 0
    iid_list = []
    ood_list1 = []
    ood_list2 = []
    ood_list3 = []
    ood_list4 = []


    size = 0
    if torch.cuda.is_available():
        size = 50000
    else:
        size = 50000
    size = 10000
    '''
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_data = np.concatenate([state_batch, action_batch], axis=1)
        iid_distance = kd_tree.query(iid_data, k=k)[0][1:]

        iid_distance = np.mean(iid_distance, axis=1, keepdims=False)

        iid_list.extend(iid_distance)


        i += batch_size

        print("step:{}, iid: {:4f}".format(i, np.mean(iid_distance)))
    '''
    print(memory.state_dim + memory.action_dim)
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_data = np.concatenate([state_batch, action_batch], axis=1)
        iid_distance = kd_tree.query(iid_data, k=k)[0][:, 1:]

        ood_action_batch1 = action_batch + 0.01 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch1 = np.clip(ood_action_batch1, -1, 1)
        ood_data1 = np.concatenate([state_batch, ood_action_batch1], axis=1)

        ood_action_batch2 = action_batch + 0.1 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch2 = np.clip(ood_action_batch2, -1, 1)
        ood_data2 = np.concatenate([state_batch, ood_action_batch2], axis=1)

        ood_action_batch3 = action_batch + 0.3 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch3 = np.clip(ood_action_batch3, -1, 1)
        ood_data3 = np.concatenate([state_batch, ood_action_batch3], axis=1)

        ood_action_batch4 = action_batch + 1.0 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch4 = np.clip(ood_action_batch4, -1, 1)
        ood_data4 = np.concatenate([state_batch, ood_action_batch4], axis=1)

        ood_distance1 = kd_tree.query(ood_data1, k=k)[0][:, :1]
        ood_distance2 = kd_tree.query(ood_data2, k=k)[0][:, :1]
        ood_distance3 = kd_tree.query(ood_data3, k=k)[0][:, :1]
        ood_distance4 = kd_tree.query(ood_data4, k=k)[0][:, :1]

        iid_distance = np.mean(iid_distance, axis=1, keepdims=False)
        ood_distance1 = np.mean(ood_distance1, axis=1, keepdims=False)
        ood_distance2 = np.mean(ood_distance2, axis=1, keepdims=False)
        ood_distance3 = np.mean(ood_distance3, axis=1, keepdims=False)
        ood_distance4 = np.mean(ood_distance4, axis=1, keepdims=False)


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

    iid_list = iid_list / (memory.state_dim + memory.action_dim)
    ood_list1 = ood_list1 / (memory.state_dim + memory.action_dim)
    ood_list2 = ood_list2 / (memory.state_dim + memory.action_dim)
    ood_list3 = ood_list3 / (memory.state_dim + memory.action_dim)
    ood_list4 = ood_list4 / (memory.state_dim + memory.action_dim)


    print("0%: ", iid_list[0])
    print("0.1%: ", iid_list[np.int32(len(iid_list)*0.001)], ood_list1[np.int32(len(ood_list1)*0.001)], ood_list2[np.int32(len(ood_list2)*0.001)], ood_list3[np.int32(len(ood_list3)*0.001)], ood_list4[np.int32(len(ood_list4)*0.001)])
    print("1%: ", iid_list[np.int32(len(iid_list)*0.01)], ood_list1[np.int32(len(ood_list1)*0.01)], ood_list2[np.int32(len(ood_list2)*0.01)], ood_list3[np.int32(len(ood_list3)*0.01)], ood_list4[np.int32(len(ood_list4)*0.01)])
    print("10%: ", iid_list[np.int32(len(iid_list)*0.1)], ood_list1[np.int32(len(ood_list1)*0.1)], ood_list2[np.int32(len(ood_list2)*0.1)], ood_list3[np.int32(len(ood_list3)*0.1)], ood_list4[np.int32(len(ood_list4)*0.1)])
    print("20%: ", iid_list[np.int32(len(iid_list)*0.2)], ood_list1[np.int32(len(ood_list1)*0.2)], ood_list2[np.int32(len(ood_list2)*0.2)], ood_list3[np.int32(len(ood_list3)*0.2)], ood_list4[np.int32(len(ood_list4)*0.2)])
    print("50%: ", iid_list[np.int32(len(iid_list)*0.5)], ood_list1[np.int32(len(ood_list1)*0.5)], ood_list2[np.int32(len(ood_list2)*0.5)], ood_list3[np.int32(len(ood_list3)*0.5)], ood_list4[np.int32(len(ood_list4)*0.5)])
    print("99%: ", iid_list[np.int32(len(iid_list)*0.99)], ood_list1[np.int32(len(ood_list1)*0.99)], ood_list2[np.int32(len(ood_list2)*0.99)], ood_list3[np.int32(len(ood_list3)*0.99)], ood_list4[np.int32(len(ood_list4)*0.99)])
    print("100%: ", iid_list[-1], ood_list1[-1], ood_list2[-1], ood_list3[-1], ood_list4[-1])

    return iid_list


def test_tree_true_sn(memory, kd_tree, feature_nn, k=3, batch_size=2560):
    i = 0
    iid_list = []
    ood_list1 = []
    ood_list2 = []
    ood_list3 = []
    ood_list4 = []


    size = 0
    if torch.cuda.is_available():
        size = 50000
    else:
        size = 50000
    size = memory.size
    '''
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_data = np.concatenate([state_batch, action_batch], axis=1)
        iid_distance = kd_tree.query(iid_data, k=k)[0][1:]

        iid_distance = np.mean(iid_distance, axis=1, keepdims=False)

        iid_list.extend(iid_distance)


        i += batch_size

        print("step:{}, iid: {:4f}".format(i, np.mean(iid_distance)))
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(memory.state_dim + memory.action_dim)
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_data = feature_nn(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(action_batch).to(device)).detach().cpu().numpy()
        iid_distance = kd_tree.query(iid_data, k=k)[0][1:2]

        #print("step:{}, iid: {:4f}".format(i, np.mean(iid_distance)))

        #i += batch_size

        ood_action_batch1 = action_batch + 0.01 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch1 = np.clip(ood_action_batch1, -1, 1)
        ood_data1 = feature_nn(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch1).to(device)).detach().cpu().numpy()

        ood_action_batch2 = action_batch + 0.1 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch2 = np.clip(ood_action_batch2, -1, 1)
        ood_data2 = feature_nn(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch2).to(device)).detach().cpu().numpy()

        ood_action_batch3 = action_batch + 0.1 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch3 = np.clip(ood_action_batch3, -1, 1)
        ood_data3 = feature_nn(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch3).to(device)).detach().cpu().numpy()

        ood_action_batch4 = action_batch + 1.0 * np.random.normal(0., 1., size=action_batch.shape)
        ood_action_batch4 = np.clip(ood_action_batch4, -1, 1)
        ood_data4 = feature_nn(torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch4).to(device)).detach().cpu().numpy()

        ood_distance1 = kd_tree.query(ood_data1, k=k)[0][:1]
        ood_distance2 = kd_tree.query(ood_data2, k=k)[0][:1]
        ood_distance3 = kd_tree.query(ood_data3, k=k)[0][:1]
        ood_distance4 = kd_tree.query(ood_data4, k=k)[0][:1]

        iid_distance = np.mean(iid_distance, axis=1, keepdims=False)
        ood_distance1 = np.mean(ood_distance1, axis=1, keepdims=False)
        ood_distance2 = np.mean(ood_distance2, axis=1, keepdims=False)
        ood_distance3 = np.mean(ood_distance3, axis=1, keepdims=False)
        ood_distance4 = np.mean(ood_distance4, axis=1, keepdims=False)


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

def test_tree_true_sns(memory, kd_trees, feature_nns, k=3, batch_size=2560):
    i = 0
    iid_list = []
    ood_list1 = []
    ood_list2 = []
    ood_list3 = []
    ood_list4 = []


    size = 0
    if torch.cuda.is_available():
        size = 50000
    else:
        size = 50000
    size = memory.size
    #size = 20000
    '''
    while i + batch_size < size:
        index = np.arange(i, i+batch_size)
        state_batch, action_batch = memory.sample_by_index(ind=index, return_np=True)
        iid_data = np.concatenate([state_batch, action_batch], axis=1)
        iid_distance = kd_tree.query(iid_data, k=k)[0][1:]

        iid_distance = np.mean(iid_distance, axis=1, keepdims=False)

        iid_list.extend(iid_distance)


        i += batch_size

        print("step:{}, iid: {:4f}".format(i, np.mean(iid_distance)))
    '''
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


        iid_distances = []
        ood_distances1 = []
        ood_distances2 = []
        ood_distances3 = []
        ood_distances4 = []
        for j in range(len(feature_nns)):
            iid_data = feature_nns[j](torch.FloatTensor(state_batch).to(device), torch.FloatTensor(action_batch).to(device)).detach().cpu().numpy()
            iid_distance = kd_trees[j].query(iid_data, k=k)[0][:, 1:]

            ood_data1 = feature_nns[j](torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch1).to(device)).detach().cpu().numpy()

            ood_data2 = feature_nns[j](torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch2).to(device)).detach().cpu().numpy()

            ood_data3 = feature_nns[j](torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch3).to(device)).detach().cpu().numpy()

            ood_data4 = feature_nns[j](torch.FloatTensor(state_batch).to(device), torch.FloatTensor(ood_action_batch4).to(device)).detach().cpu().numpy()

            ood_distance1 = kd_trees[j].query(ood_data1, k=k)[0][:, :1]
            ood_distance2 = kd_trees[j].query(ood_data2, k=k)[0][:, :1]
            ood_distance3 = kd_trees[j].query(ood_data3, k=k)[0][:, :1]
            ood_distance4 = kd_trees[j].query(ood_data4, k=k)[0][:, :1]

            iid_distance = np.mean(iid_distance, axis=1, keepdims=True)
            ood_distance1 = np.mean(ood_distance1, axis=1, keepdims=True)
            ood_distance2 = np.mean(ood_distance2, axis=1, keepdims=True)
            ood_distance3 = np.mean(ood_distance3, axis=1, keepdims=True)
            ood_distance4 = np.mean(ood_distance4, axis=1, keepdims=True)

            iid_distances.append(iid_distance)
            ood_distances1.append(ood_distance1)
            ood_distances2.append(ood_distance2)
            ood_distances3.append(ood_distance3)
            ood_distances4.append(ood_distance4)


        iid_list.extend(np.max(np.concatenate(iid_distances, axis=1), axis=1))
        ood_list1.extend(np.max(np.concatenate(ood_distances1, axis=1), axis=1))
        ood_list2.extend(np.max(np.concatenate(ood_distances2, axis=1), axis=1))
        ood_list3.extend(np.max(np.concatenate(ood_distances3, axis=1), axis=1))
        ood_list4.extend(np.max(np.concatenate(ood_distances4, axis=1), axis=1))

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


    print(np.sum((ood_list3 < ood_list2[np.int32(len(ood_list1)*0.5)]).astype(float)) / len(ood_list3), abnormal[2] / len(ood_list3))
    print(np.sum((ood_list4 < ood_list2[np.int32(len(ood_list1)*0.5)]).astype(float)) / len(ood_list4), abnormal[3] / len(ood_list4))

    print(abnormal)

    return iid_list
