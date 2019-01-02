import tensorflow as tf

'''
	按照张量的方式来遍历读取数据
'''
input_data = [1,2,3,5,8]
# 1.定义数据集的构造方法：从张量中构建、从文件中构建
dataset = tf.data.Dataset.from_tensor_slices(input_data)

# 2.定义遍历器来遍历数据集
iterator = dataset.make_one_shot_iterator()
# 3.从遍历器中读取数据，作为计算图其他部分的输入
x = iterator.get_next()
y = x * x

with tf.Session() as sess:
	for i in range(len(input_data)):
		print(sess.run(y))

'''
	按照文本文件的方式来遍历读取数据(自然语言处理任务)
'''
input_files = ['path/to/file1', 'path/to/file2']
dataset = tf.data.TextLineDataset(input_files)

iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()

with tf.Session() as sess:
	for i in range(3): print(sess.run(x))


'''
	按照TFRecord方式来遍历数据(图像相关任务)，每个TFRecord都有自己不同的
	feature格式，因此在读取TFRecord时，需要提供一个parser函数来解析读取
	的TFRecord格式数据
'''
def parse(record):
	"""
	解析读入的一个样例
	:param record: 解析得到的样例
	:return: 返回解析后的数据
	"""
	features = tf.parse_single_example(
		record,
		features={
			'img': tf.FixedLenFeature([], tf.int64),
			'label': tf.FixedLenFeature([], tf.int64),
		})
	return features['img'], features['lable']

input_tfrecords = ['/path/to/tfrecord1', '/path/to/tfrecord2']
dataset = tf.data.TFRecordDataset(input_files)
# map()函数表示对数据集中每一条数据调用相应的方法
# 使用TFRecordDataset读出的是二进制数据，这里需要通过map函数调用parse函数
# 对二进制文件进行解析
# map函数可以用来完成其他数据预处理的工作
dataset = dataset.map(parse)

# 使用该函数，数据集的所有参数必须已经确定了
iterator = dataset.make_one_shot_iterator()
img, label = iterator.get_next()

with tf.Session() as sess:
	for i in range(10): print(sess.run([img, label]))

'''
	动态初始化数据集的栗子
'''
# 样例解析函数一致，不重复写了

input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parse)

iterator = dataset.make_initializable_iterator()
img, label = iterator.get_next()

with tf.Session() as sess:
	# 初始化iterator，并给出input_files的值
	sess.run(iterator.initializer, feed_dict={
		input_files:['/path/to/tfrecord1','/path/to/tfrecord2']
	})
	# 遍历所有数据一个batch
	while True:
		try:
			sess.run([img, label])
		except tf.errors.OutOfRangeError:
			break