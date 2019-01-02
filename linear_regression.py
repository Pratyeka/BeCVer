import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

learning_rate = 0.01
training_epoch = 1000
display_step = 100

train_x = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                      7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                      2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_x.shape[0]

# placeholder表示占位符，可以在每次需要图像时再load数据
# 避免了一次加载全部的数据或者每个batch数据创建新节点的尴尬
X = tf.placeholder('float')
Y = tf.placeholder('float')
W = tf.Variable(rng.randn(), name='weight')
B = tf.Variable(rng.randn(), name='bias')

pred = tf.add(tf.multiply(X, W), B)

cost = tf.reduce_sum(tf.pow(pred-Y,2)) / (2*n_samples)
# 定义优化器函数，指定优化的损失函数
opter = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# 以上的变量名都表示一个引用，没有实际的数字，
# 在实际使用时需要在会话上下文中使用sess.run函数来得到计算结果

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoch):
        for x,y in zip(train_x, train_y):
            sess.run(opter, feed_dict={X:x, Y:y})
        if not (epoch+1) % display_step:
            c = sess.run(cost, feed_dict={X:train_x, Y:train_y})
            print('epoch {} cose == {}'.format(epoch+1, c))

    print('w = {}, b = {}'.format(sess.run(W), sess.run(B)))
    plt.plot(train_x, train_y, 'ro', label='Original Data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(B), label='Fitted Line')
    plt.plot(train_x, sess.run(train_x * W + B), 'bo', label='Another Fitted Line')
    plt.legend()
    plt.show()
