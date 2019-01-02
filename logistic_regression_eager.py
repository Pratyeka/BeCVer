from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.enable_eager_execution()
tfe = tf.contrib.eager

learning_rate = 0.1
batch_size = 128
num_steps = 10000
display_step = 100

# eager模式下，输入数据不是placeholder模式，需要直接给定数据
mnist = input_data.read_data_sets('D:/Job/DataSets/mnist', one_hot=False)
# 根据指定的输入，返回对应切片的dataset实例对象
dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels))
# repeat函数指定数据集被重复的次数，None时无限循环；batch函数将数据集的连续元素组合成batch；
# prefetch函数从数据集中提前获取一批数据
dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
# 将dataset变为一个可迭代对象
dataset_iter = tfe.Iterator(dataset)

W = tfe.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=(784,10)),name='W')
b = tfe.Variable(initial_value=tf.random_normal(dtype=tf.float32, shape=(10,)), name='b')

# 前向传播收尾函数定义
# eager模式下，预测结果、代价值都是以函数形式给出，普通模式下是按照节点形式给出
pred = lambda x:tf.matmul(x, W) + b
# 交叉熵损失函数
loss = lambda x,y:tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred(x), labels=y))
# label是单值模式，tf.equal函数可以得到bool的结果，cast函数将bool结果转为数值，统计得到准确率
accuracy = lambda x,y:tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(pred(x)), 1), y), tf.float32))

# 优化器定义
# eager模式下，优化器实例定义，手动反向传播；普通模式下指定对应的损失函数，自动计算
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# 梯度应用实例，接受的参数是loss函数的参数，‘猜测是利用偏函数的方式补全了实例对象的其他参数’
# 函数式编程，将函数作为一等公民
grad = tfe.implicit_gradients(loss)

# 模型训练
average_loss = 0.
average_acc = 0.

for step in range(num_steps):
	d = dataset_iter.next()  # 数据获取
	x_batch = d[0]
	y_batch = tf.cast(d[1], dtype=tf.int64)  # 数据类型转换

	batch_loss = loss(x_batch, y_batch)
	average_loss += batch_loss
	batch_accuracy = accuracy(x_batch, y_batch)
	average_acc += batch_accuracy

	if step == 0: print('Init loss = {}'.format(average_loss))
	# 优化器开始工作，反向传播算法调整参数，
	optimizer.apply_gradients(grad(x_batch, y_batch))

	if (step+1) % display_step == 0 or step == 0:
		if step > 0:
			average_loss /= display_step
			average_acc /= display_step
		print('step:{}, loss:{}, accuracy:{}'.format(step, average_loss, average_acc))
		average_loss = 0.
		average_acc = 0.

testX, testY = mnist.test.images, mnist.test.labels
test_acc = accuracy(testX, testY)
print('Test Accuracy:{}'.format(test_acc))