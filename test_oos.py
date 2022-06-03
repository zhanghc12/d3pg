import numpy as np

len_data = 100
data = np.random.normal(np.zeros(len_data), np.ones(len_data))

print(data)

mean = np.mean(data)
std = np.mean((data - mean)**2)

print(mean)
print(std)

emp_data = np.random.normal([mean] * len_data, [std] * len_data)
print(emp_data)

# out-of-sample rate: the pdf < 1e-3
oos_count = 0
for d in emp_data:
    if np.min((d - data) ** 2) > 1e-4:
        oos_count += 1

# 0.64, 0.57, 0.55, 0.62, 0.67, -.6
print('oos_ratio', oos_count / len_data)
# out_of_sample_ratio =
print(np.mean([64, 57, 55, 62, 67, 48, 57, 58, 58]))
print(np.std([64, 57, 55, 62, 67, 48, 57, 58, 58]))