import numpy as np

d1={'key1':np.array([50,50]), 'key2':[50,100]}
np.save("/tmp/d1.npy", d1)
d2=np.load("/tmp/d1.npy", allow_pickle=True)

print(d1['key1'])
print(d1.get('key1'))
print(d2.item().get('key2'))


array = np.zeros([20])
for a in array:
    print(a)