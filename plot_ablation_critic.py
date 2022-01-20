import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
# 获取数据
titanic = sns.load_dataset("titanic")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

#sns.set()
#current_palette = sns.color_palette()
#sns.palplot(current_palette)
def to_percent(temp, position):
    return '%1.0f'%(1*temp) + '%'

fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))

fontsize = 30

plt.subplot(1,2,1)

data = [60, 20, 10, 40]
labels = [1, 50, 100, 200]
plt.plot(labels, data, linewidth=5, ms='30', marker='.')
#plt.scatter(labels, data, marker='x', linewidth=5)

plt.ylim(0, 100)
plt.ylabel('Performance', fontsize=fontsize)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Halfcheetah medium', fontsize=fontsize)

plt.subplot(1,2,2)
data = [60, 20, 10, 40]
labels = [1, 50, 100, 200]
plt.plot(labels, data, linewidth=5, ms='30', marker='.')
#plt.scatter(labels, data, marker='x', linewidth=5)

plt.ylim(0, 100)
# plt.ylabel('Percentage Difference', fontsize=fontsize)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Halfcheetah medium-replay', fontsize=fontsize)

plt.show()

'''
size = 5
x = np.arange(size)
cheetah1 = [60.1, 49.0, 40.8, 40.9, 29.0]
cheetah2 = [63.6, 52.2, 59.6, 59.0, 44.8]

walker1 = [88.5, 69.7, 54.7, 45.9, 1.7]
walker2 = [106.6, 97.0, 105.2, 70.0, 65.8]

hopper1 = [111.1, 109.0, 109.9, 109.5, 102.0]
hopper2 = [111.5, 111.1, 111.4, 110.9, 111.2]

fontsize = 22
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.subplot(1,2,1)
plt.bar(x, cheetah1,  width=width, label='CQL', color='#00917F')
plt.bar(x + width, cheetah2, width=width, label='Our method', color='#CE2D05')
# plt.bar(x + 2 * width, c, width=width, label='c')
plt.xlabel('Random ratio', fontsize=fontsize)
plt.ylabel('Normalized Score', fontsize=fontsize)
plt.title('Halfcheetah', fontsize=fontsize)
#axes[0].xaxis.set_ticks(np.array(['0.5', '0.6', '0.7', '0.8', '0.9'], minor=True)
plt.xticks(x, ('50%', '60%', '70%', '80%', '90%'))
# plt.xticks(y, ('50%', '60%', '70%', '80%', '90%'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.subplot(1,2,2)

plt.bar(x, walker1,  width=width, label='CQL', color='#00917F')
plt.bar(x + width, walker2, width=width, label='Our method', color='#CE2D05')
plt.xlabel('Random Ratio', fontsize=fontsize)
plt.title('Walker2d', fontsize=fontsize)
plt.xticks(x, ('50%', '60%', '70%', '80%', '90%'))

plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

line_labels = ['CQL', 'Our method']
fig.legend(line_labels, fontsize=22, loc="upper center", ncol=2)
'''
# plt.legend(loc='upper center')
# plt.show()