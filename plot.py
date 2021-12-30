from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set()#style="dark", palette="muted", color_codes=True)
fig, axes = plt.subplots(1, 1, figsize=(10, 6.5))
le = []

env = 'HalfCheetah'
line_labels = ['HalfCheetah-v2']
fontsize=22
colorplus = ["#2CAFAC"]#, "#C11CAD", "#8338EC"]#, M2PG":"#FF006E"}

# dirname = '/Users/peixiaoqi/aaai/ablation/tf_logs_0828/GridWorld-v0/19/'
dirname = '/data/zhanghc/d3pg/12_23/'
version_names = [dirname + '2021-12-30_02-16-31_Dueling_' + env]

filenames = []
for l1 in os.listdir(dirname):
    if l1.startswith('2021-12-30_02-16-31_Dueling_' + env):
        for l2 in os.listdir(dirname + l1 + '/'):
            filenames.append(dirname + l1 + '/' + l2)
print(filenames)

color = 'blue'
shade_colors = 'lightblue'
average_num = 30

seg_data = []
for file_name in filenames:
    ea = event_accumulator.EventAccumulator(file_name)
    ea.Reload()
    val_psnr = ea.scalars.Items('test/return')
    seed_data_value = []
    for i in val_psnr:
        seed_data_value.append(i.value)
    seed_data_value = np.convolve(seed_data_value, np.ones(average_num) / average_num, mode='valid')
    seg_data.append(seed_data_value)


mean_data = []
std_data = []

for i in range(len(seg_data[0])):
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
l1 = axes.plot(range(len(mean_data)), mean_data, color=color)[0]#, label=env+'-'+ratios[k])
print(len(mean_data), len(lower_data), len(upper_data))
axes.fill_between(range(len(lower_data)),lower_data, upper_data, color=shade_colors, alpha=0.15)
le.append(l1)
# base_data[i] = seg_data

# base_data[i] = seg_data
axes.set_xlabel('Iterations', fontsize=fontsize)
axes.set_ylabel('Return', fontsize=fontsize)
# axes[1].set_title('Eta', fontsize=fontsize)


axes.legend(le, line_labels, fontsize=22, loc="upper left", ncol=1)
fig.savefig('beta_grid.png', bbox_inches='tight')
# plt.show()




