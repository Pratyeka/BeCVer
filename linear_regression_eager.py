from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()
tfe = tf.contrib.eager

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                      7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                      2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf的Variable不支持eager模式，所以只能用tfe
W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())

# 定义线性计算
lreg = lambda x: x*W+b
# 定义损失函数
cost = lambda x,y: tf.reduce_sum(tf.pow(lreg(x)-y,2) / (2*x.shape[0]))

# 定义优化器，这里的优化器需要手动的计算 损失梯度等信息
opter = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 梯度计算函数
grad = tfe.implicit_gradients(cost)

print('Initial cost = {}, W = {}, b = {}'.format(cost(train_X, train_Y), W, b))

for _ in range(1000):
	# 优化器根据梯度值更新参数
	opter.apply_gradients(grad(train_X, train_Y))
	if not (_+1)%100:
		print('Epoch {} cost = {}, w = {}, b = {}'.format(_+1, cost(train_X, train_Y), W, b))

plt.plot(train_X, train_Y, 'ro', label='Original Data')
plt.plot(train_X, np.array(train_X * W + b), label='Fitted Line')
plt.legend()
plt.show()