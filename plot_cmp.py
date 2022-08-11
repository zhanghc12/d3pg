import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="")
fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
fontsize = 30

plt.subplot(1,2,1)
plt.ylim(0, 35)
plt.xlabel('Noise Scale', fontsize=fontsize)
plt.ylabel('Uncertainty Scale', fontsize=fontsize)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('hopper-medium-replay', fontsize=fontsize)

x = [0.01, 0.10, 0.30, 1.00]
x1 = [1,1,1,1.01]
x2 = [1,1,1,2.00]
x3 = [1,1,1.06,3.33]
x4 = [1,9.23, 15.31, 26.69]
plt.plot(x, x1, color='crimson', linestyle='-', marker='*', linewidth=5, ms=20)
plt.plot(x, x2, color='royalblue', linestyle='-',  marker='*', linewidth=5, ms=20)
plt.plot(x, x3, color='darkorange', linestyle='-',  marker='*', linewidth=5, ms=20)
plt.plot(x, x4, color='g', linestyle='-', marker='*', linewidth=5, ms=20)

line_labels = ['MC  Dropout', 'Ensemble', 'VAE', 'Our method']
plt.legend(line_labels, fontsize=22, ncol=1)

plt.subplot(1,2,2)
plt.ylim(0, 35)
plt.xlabel('Noise Scale', fontsize=fontsize)
plt.ylabel('Uncertainty Scale', fontsize=fontsize)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('hopper-expert', fontsize=fontsize)

x = [0.01, 0.10, 0.30, 1.00]
x1 = [1,1,1,1]
x2 = [1,1,1,3.25]
x3 = [1,1.13,2.06,6.81]
x4 = [1,9.26, 19.63,30.37]
plt.plot(x, x1, color='crimson', linestyle='-', marker='*', linewidth=5, ms=20)
plt.plot(x, x2, color='royalblue', linestyle='-',  marker='*', linewidth=5, ms=20)
plt.plot(x, x3, color='darkorange', linestyle='-',  marker='*', linewidth=5, ms=20)
plt.plot(x, x4, color='g', linestyle='-', marker='*', linewidth=5, ms=20)

plt.show()