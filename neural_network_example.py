import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('D:/Job/DataSets/mnist', one_hot=True)

learning_rate = 0.01
num_steps = 10000
batch_size = 128
display_num = 200

# 在tf中，如果将数据数据使用constant来表示，每个batch的数据都会创建一个节点，这样造成计算图过大，且利用率很低
# 因此使用placeholder机制：
# placeholder用来提供数据，相当于一个占位节点，该节点的数据在程序运行时才会被指定
# placeholder定义时需要指定其数据类型，且类型不可变
# placeholder中数据尺寸可以在计算中推断得出，因此非必须设定的参数
# 程序运行时，使用feed_dict字典形式来给出实际数据
mnist_train_img = tf.placeholder('float', shape=[None, 784], name='train_img')
mnist_train_label = tf.placeholder('float', shape=[None, 10], name='train_label')

W1 = tf.Variable(tf.random_normal([784, 256], mean=0.2, stddev=1.1), name='W1')
b1 = tf.Variable(tf.random_normal([256], mean=0.1, stddev=1.2), name='b1')
W2 = tf.Variable(tf.random_normal([256, 256]), name='W2')
b2 = tf.Variable(tf.random_normal([256]), name='b2')
Ow = tf.Variable(tf.random_normal([256, 10]), name='Ow')
Ob = tf.Variable(tf.random_normal([10]), name='Ob')

def nerual_net(x):
	layer1 = tf.add(tf.matmul(x, W1) + b1)
	layer2 = tf.add(tf.matmul(layer1, W2) + b2)
	out_layer = tf.matmul(layer2, Ow) + Ob
	return out_layer

logits = nerual_net(mnist_train_img)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=mnist_train_label))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(mnist_train_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(1, num_steps+1):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		sess.run(train_op, feed_dict={mnist_train_img:batch_x, mnist_train_label:batch_y})
		if step % display_num == 0 or step == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict=
			{mnist_train_img:batch_x, mnist_train_label:batch_y})
	print('Test acc:{}'.format(sess.run(accuracy, feed_dict={mnist_train_img:mnist.test.images,
	                                                         mnist_train_label:mnist.test.labels})))