import d4rl
import gym
from sklearn.manifold import TSNE
import time
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import seaborn as sns

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    strcolors = []

    for i in range(len(colors)):
        if colors[i] == 0:
            strcolors.append('r')
        if colors[i] == 1:
            strcolors.append('#00917F')
        if colors[i] == 2:
            strcolors.append('#0099CC')
        if colors[i] == 3:
            strcolors.append('y')
    #'''

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=strcolors)# palette[colors.astype(np.int)])

    ax.set_facecolor('#002E71')
    ax.grid(False)

    # ax.axis('off')
    # ax.axis('tight')
    # plt.savefig(env_name)

    plt.show()

env_name = 'hopper-expert-v0'

eval_env = gym.make(env_name)
dataset = d4rl.qlearning_dataset(eval_env)

mean_obs = np.mean(dataset['observations'], axis=0)
std_obs = np.std(dataset['observations'], axis=0)
mean_actions = np.mean(dataset['actions'], axis=0)
actions = dataset['actions']
post_observations = (dataset['observations'] - mean_obs) / (std_obs + 1e-5)


# build new datasets
# random and expert

ood_action_batch1 = actions + 0.01 * np.random.normal(0., 1., size=actions.shape)
ood_action_batch1 = np.clip(ood_action_batch1, -1, 1)
ood_action_batch2 = actions + 0.1 * np.random.normal(0., 1., size=actions.shape)
ood_action_batch2 = np.clip(ood_action_batch2, -1, 1)
ood_action_batch3 = actions + 0.3 * np.random.normal(0., 1., size=actions.shape)
ood_action_batch3 = np.clip(ood_action_batch3, -1, 1)
ood_action_batch4 = actions + 1.0 * np.random.normal(0., 1., size=actions.shape)
ood_action_batch4 = np.clip(ood_action_batch4, -1, 1)

x0 = np.concatenate([post_observations, actions], axis=1)
x1 = np.concatenate([post_observations, ood_action_batch1], axis=1)
x2 = np.concatenate([post_observations, ood_action_batch2], axis=1)
x3 = np.concatenate([post_observations, ood_action_batch3], axis=1)
x4 = np.concatenate([post_observations, ood_action_batch4], axis=1)

labels0 = np.zeros([len(x0), 1], dtype=np.int32)
labels1 = np.zeros([len(x0), 1], dtype=np.int32) + 1
labels2 = np.zeros([len(x0), 1], dtype=np.int32) + 2
labels3 = np.zeros([len(x0), 1], dtype=np.int32) + 3
labels4 = np.zeros([len(x0), 1], dtype=np.int32) + 4

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

shade = False

time_start = time.time()

indices = np.arange(len(x))
np.random.shuffle(indices)
indices = indices[:100]
fashion_tsne = TSNE(random_state=RS, perplexity=50, learning_rate=10).fit_transform(x[indices])
print( 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fashion_scatter(fashion_tsne, base_labels[indices])







