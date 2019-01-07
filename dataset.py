# 对图像进行数据预处理可以尽可能的减小预处理对于训练速度的影响
# 为了减少复杂预处理给模型训练速度带来的影响，TF提供了多线程处理输入数据的解决方案

# 不同格式、属性的数据会对模型的输入造成困扰，
# TF提供了TFRecord格式来统一不同的原始数据格式，可以更加有效的管理不同的属性

### TFRecord格式统一存储数据
"""
TFRecord文件中的数据都是通过tf.train.Example Protocol Buffer的格式存储的
tf.train.Example中包含了一个从属性到取值的字典：其中属性是一个字符串，对应的取值是如下一种：
           1.BytesList    2.FloatList     3.Int64List
"""
# mnist数据存为TFRecord格式
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('D:/Job/DataSets/mnist', one_hot=True)
# 因为取值有要求，所以这里做格式统一处理
# 生成整数型取值列表
_int64_feature = lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成浮点型取值列表
_bytes_feature = lambda value: tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]

num_examples = mnist.train.num_examples
filename = '.'

# TFRecord的写数据接口
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
	image_raw = images[index].tostring()
	# 按照tf.train.Example格式对每个数据进行处理
	example = tf.train.Example(features=tf.train.Feature(feature={
		'pixels': _int64_feature(pixels),
		'label': _int64_feature(labels[index]),
		'images': _bytes_feature(image_raw)
	}))
	# 调用TFRecord实例的writer方法，将example格式的数据存入
	writer.write(example.SerializeToString())
writer.close()

## TFRecord数据读取接口
reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer(['path/to/output.tfrecord'])

# 从文件中读出一个样例，也可使用read_up_to函数一次性读取多个样例
_, serialized_example = reader.read(filename_queue)
# 解析读入的样例，如果需要解析多个样例，可以使用函数parse_example
# 使用tf.FixedLenFeature解析结果为Tensor
# 使用tf.VarLenFeature解析结果为SparseTensor
# 解析数据的格式需要和写入数据的格式一致
features = tf.parse_single_example(serialized_example,features={
	'images': tf.FixedLenFeature([], tf.string),
	'label': tf.FixedLenFeature([], tf.int64),
	'pixels': tf.FixedLenFeature([], tf.int64),
})

images = tf.decode_raw(features['images'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixel = tf.cast(features['pixels'], tf.int32)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	for i in range(10):
		print(sess.run([images, label, pixels]))