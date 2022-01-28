import os

from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn
import numpy as np


sns.set()#style="dark", palette="muted", color_codes=True)
fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
axes = axes.reshape(-1)
le = []
# line_labels = ['1.0', '10.0', '100.0', '1000.0']#, '0.9']
line_labels = ['0.1', '1', '10', '100']#, '0.9']

fontsize=22
colorplus = ["#2CAFAC", "#FB5607", "#1982C4", "#C11CAD"]#, "#8338EC"]#, M2PG":"#FF006E"}


# 2022-01-22_09-26-22_TQC_halfcheetah-medium-replay-v2_s1_ver3_thre0.1_tau0.005_d200_n5_bs100.0_od6
# all the find the dirname:

env2 = 'hopper-medium-replay-v2'
env_name = 'halfcheetah-medium-replay'

env = 'halfcheetah-medium-replay-v2'
env2 = 'walker2d-medium-expert-v2'
env_name2 = 'walker2d-medium-expert'

dirname = '/Users/peixiaoqi/icml2022/ablation_actor/'
# env_name = env + '-v2'
# ratios = ['bs1.0', 'bs10.0', 'bs100.', 'bs0.8', 'bs0.9']
ratios = ['bs0.1', 'bs1.0', 'bs10.0', 'bs100.0']

all_names = [[], [], [], []]
# all_names = [[], []]

for i, ratio in enumerate(ratios):
    for l1 in os.listdir(dirname):
        if env in l1 and ratio in l1:
            all_names[i].append(dirname + '/' + l1)

print(len(all_names[0]))
print(len(all_names[1]))

total_length = 800
average_num = 30
print(all_names)
for k, all_names_per_version in enumerate(all_names):
    seg_data = []
    for file_name in all_names_per_version:
        ea = event_accumulator.EventAccumulator(file_name)
        ea.Reload()
        val_psnr = ea.scalars.Items('test/d4rl_score')
        if '2022-01-22_17-59-12_TQC_halfcheetah-medium-replay-v2_s0_ver1_thre0.1_tau0.005_d200_n5_bs1.0_od6' in file_name:
            break
        seed_data_value = []
        #print(file_name)

        for i in val_psnr:
            seed_data_value.append(i.value)
            #print(i.value)

        seed_data_value = np.convolve(seed_data_value, np.ones(average_num) / average_num, mode='valid')
        seg_data.append(seed_data_value)

    mean_data = []
    std_data = []

    # print(len(seg_data[0]))
    for i in range(total_length):
    # for i in range(len(seg_data[0])):
        print(i)

        vals = []
        for j in range(len(seg_data)):
            vals.append(seg_data[j][i])

        mean = np.mean(vals)
        std = np.sqrt(np.var(vals))
        mean_data.append(mean)
        std_data.append(std)
    print(np.mean(mean_data[-50:]))
    upper_data = np.array(mean_data) + np.array(std_data)
    lower_data = np.array(mean_data) - np.array(std_data)

    # plt.plot(base_data[0][0], mean_data)
    # plt.fill_between(base_data[0][0], lower_data, upper_data, color='lightblue',alpha=0.25)
    l1 = axes[0].plot(range(len(mean_data)), mean_data, color=colorplus[k])[0]#, label=env+'-'+ratios[k])
    print(len(mean_data), len(lower_data), len(upper_data))
    axes[0].fill_between(range(len(lower_data)),lower_data, upper_data, color=colorplus[k], alpha=0.15)
    le.append(l1)

axes[0].set_xlabel('Iterations', fontsize=fontsize)
axes[0].set_ylabel('Return', fontsize=fontsize)
axes[0].set_title(env_name, fontsize=30)



#env2 = 'halfcheetah-medium-replay-v2'

dirname = '/Users/peixiaoqi/icml2022/ablation_actor/'
# env_name = env2 + '-v2'

all_names = [[], [], [], []]#, []]


for i, ratio in enumerate(ratios):
    for l1 in os.listdir(dirname):
        if env2 in l1 and ratio in l1:
            all_names[i].append(dirname + '/' + l1)

average_num = 30
print(all_names)

for k, all_names_per_version in enumerate(all_names):
    seg_data = []
    for file_name in all_names_per_version:
        ea = event_accumulator.EventAccumulator(file_name)
        ea.Reload()
        print(file_name)
        try:
            val_psnr = ea.scalars.Items('test/d4rl_score')
        except:
            break
        seed_data_value = []
        for i in val_psnr:
            seed_data_value.append(i.value)
        seed_data_value = np.convolve(seed_data_value, np.ones(average_num) / average_num, mode='valid')
        seg_data.append(seed_data_value)

    mean_data = []
    std_data = []

    for i in range(total_length): #len(seg_data[0])):
        vals = []
        for j in range(len(seg_data)):
            vals.append(seg_data[j][i])

        mean = np.mean(vals)
        std = np.sqrt(np.var(vals))
        mean_data.append(mean)
        std_data.append(std)
    print(np.mean(mean_data[-50:]))
    upper_data = np.array(mean_data) + np.array(std_data)
    lower_data = np.array(mean_data) - np.array(std_data)

    # plt.plot(base_data[0][0], mean_data)
    # plt.fill_between(base_data[0][0], lower_data, upper_data, color='lightblue',alpha=0.25)
    axes[1].plot(range(len(mean_data)), mean_data, color=colorplus[k])[0]# , label=env+'-'+ratios[k])
    print(k)
    print(len(mean_data), len(lower_data), len(upper_data))
    axes[1].fill_between(range(len(lower_data)),lower_data, upper_data, color=colorplus[k], alpha=0.15)

    # base_data[i] = seg_data
axes[1].set_xlabel('Iterations', fontsize=fontsize)
axes[1].set_ylabel('Return', fontsize=fontsize)
axes[1].set_title(env_name2, fontsize=30)


fig.legend(le, line_labels, fontsize=22, loc="upper center", ncol=5)
fig.savefig('all_random_expert_result.png', bbox_inches='tight')
plt.show()




