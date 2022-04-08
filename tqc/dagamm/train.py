from uncertainty_demo import darl2
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir='./kdd_cup.npz'
data = np.load(data_dir, allow_pickle=True)

labels = data["kdd"][:, -1]
features = data["kdd"][:, :-1]
# In this case, "atack" has been treated as normal data as is mentioned in the paper
normal_data = features[labels == 0]
normal_labels = labels[labels == 0]

n_train = int(normal_data.shape[0] * 0.5)
ixs = np.arange(normal_data.shape[0])
np.random.shuffle(ixs)
normal_data_test = normal_data[ixs[n_train:]]
normal_labels_test = normal_labels[ixs[n_train:]]


train_data = normal_data[ixs[:n_train]]
train_label = normal_labels[ixs[:n_train]]
anomalous_data = features[labels == 1]

print(len(normal_data_test), len(anomalous_data))

normal_data_test = normal_data_test[0:1000]
anomalous_data = anomalous_data[0:1000]

# anomalous_labels = labels[labels == 1][0:1000]
test_data = np.concatenate((anomalous_data, normal_data_test), axis=0)
# test_label = np.concatenate((anomalous_labels, normal_labels_test), axis=0)

print(data)
''
# now get the in distribution

feature_nn, tree = darl2.build_tree(train_data)
print('tree built')
with torch.no_grad():
    confidence = darl2.get_uncertainty(test_data, feature_nn, tree)
print(confidence)

np.save('./tqc/dagamm/confidence.npy',confidence)
# get confidence, then predict confidence
# then find the best f1 or other thing
# z = confidence.reshape(xx.shape)
# z = 1 - confidence.clip(0, 0.2)

