import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('D:/Job/DataSets/mnist', one_hot=True)

learning_rate = 0.01
train_epoch = 1000
batch_size = 100
display_step = 1

image = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784,10], mean=0.1, stddev=1))
b = tf.Variable(tf.random_normal([10], mean=0, stddev=1))

# 模型推理结果
# matmul:矩阵乘法、 multiply:对应元素的乘法
pred = tf.nn.softmax(tf.matmul(image, W) + b)

# 模型代价计算
cost = tf.reduce_mean(-tf.reduce_sum(label * tf.log(pred), reduction_indices=1))
# 模型优化器定义
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(train_epoch):
		total_batch = int(mnist.train.num_examples / batch_size)
		for i in range(total_batch):
			x, y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={image:x, label:y})
		print('epoch {} cost == {}'.format(epoch, c))

	correct_predicition = tf.equal(tf.argmax(pred, 1), tf.arg_max(label, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
	print('accuracy={}'.format(sess.run(accuracy, feed_dict={image:mnist.test.images[:3000], label:mnist.test.labels[:3000]})))