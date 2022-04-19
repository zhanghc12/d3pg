import tensorflow as tf

sess = tf.Session()
a = tf.ones([3,5])
#a = tf.ones([1, 2])
b = a * a
# b = tf.reduce_sum(a * a, axis=1)

print(sess.run(a)) # [None]
print(sess.run(b)) # [None]

# b = tf.ones([3, 1])
g1 = tf.gradients(b, a, grad_ys=1 * tf.ones([3,5]))

# g1 = tf.gradients([b], [a])
print(sess.run(g1)) # [None]
