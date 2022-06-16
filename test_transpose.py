import numpy as np
import cv2
a = np.array([[1,2],[3,4]])
b = np.transpose(a)
print(id(a))
print(id(b))
import matplotlib.image as mping

a = np.random.randint(0, 255, [3, 180, 240])
b = np.transpose(a, axes=[2,1,0])
c = np.transpose(a, axes=[1,2,0])

print(c.shape)
# mping.imsave('a.png',a)
# cv2.save()