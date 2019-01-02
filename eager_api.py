from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

print('Setting Eager mode:')
tf.enable_eager_execution()   #设置允许使用eager机制
tfe = tf.contrib.eager

print('进入eager模式, 没有sess机制了：-----------------')
print('Define constant tensors:')
a = tf.constant(2)
b = tf.constant(3)
print('a={}, b={}'.format(a,b))

print('实验基本运算操作：-----------------------------')
c = a + b
d = a * b
print('c={}(a+b), d={}(a*b)'.format(c,d))
print('c={}(a+b), d={}(a*b)'.format(a+b, a*b))


print('Compatibility with Numpy: +++++++++++++++++++++')
a = tf.constant([[2,1],[1,0]], dtype=tf.float32)
b = np.array([[2,1],[1,0]], dtype=np.float32)
print('tf constant a = {}, type is {}'.format(a, type(a)))
print('np ndarray b = {}, type is {}'.format(b, type(b)))

print('eagerTensor op ndarray, result is eagerTensor')
tf_np_add = a + b
tf_np_mul = a * b
print('eagerTensor + ndarray( a + b = {}), type is {}'.format(tf_np_add, type(tf_np_add)))
print('eagerTensor * ndarray( a * b = {}), type is {}'.format(tf_np_mul, type(tf_np_mul)))
print('eagerTensor + eagerTensor = {}'.format(tf_np_add + tf_np_mul))

print('--------------------------------------------------------')
print('iter output the ele of eagerTensor')
shape = tf_np_add.shape
print([tf_np_add[i][j] for i in range(shape[0]) for j in range(shape[1])])