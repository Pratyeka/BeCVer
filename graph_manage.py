import tensorflow as tf

# tf会自动维护一个计算图，没有指定计算图的情况下，
# 会将节点加入到默认计算图中
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
result = a + b

# 打印计算图信息，默认计算图以及计算节点所在的计算图
print(tf.get_default_graph())
print(a.graph, b.graph, result.graph)

# 不同计算图上的张量和运算不会共享

# 新建计算图g1，在g1中添加计算节点v
g1 = tf.Graph()
with g1.as_default():
	v = tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
	# get_variable函数可以控制新建变量还是获取变量值
	v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer)

with tf.Session(graph=g1) as sess:
	tf.global_variables_initializer().run()
	# reuse变量表明可以获取变量值不用新建
	with tf.variable_scope('', reuse=True):
		print(sess.run(tf.get_variable('v')))

with tf.Session(graph=g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope('', reuse=True):
		print(sess.run(tf.get_variable('v')))

# 计算图不仅用来隔离张量和计算，还提供了管理张量和计算的机制
# 使用tf.Graph.device函数指定运行计算的设备
with g1.as_default():
	a = tf.get_variable('a', shape=[1], initializer=tf.zeros_initializer)
	b = tf.get_variable('b', shape=[1], initializer=tf.ones_initializer)
	with g1.device('/cpu:0'):
		result = a + b

with tf.Session(graph=g1) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope('', reuse=True):
		print(sess.run(result))