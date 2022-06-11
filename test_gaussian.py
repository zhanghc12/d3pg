import numpy as np

a = np.random.normal(np.zeros([1000]), 0.8* np.ones([1000]))
print(np.mean(np.abs(a)))

# 0.8
# 0.1 * 0.8 -> 0.08 -> 0.064 loss
# 基本上 loss -> 1e-4, 0.01左右每一个s，但是scale之后就不知道了。