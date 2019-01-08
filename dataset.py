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


## 图像处理函数
'''
1、图像编码处理：图像存储不是以矩阵形式处理，存储的是经过压缩编码之后的结果。读取时需要把它解码成三维矩阵
   tf.gfile.FastGFile('path/to/img', 'r').read() 类似文件的读取方式获取数值
   tf.image.decode_jpeg / tf.image.decode_png
   tf.gfile.FastGFile('path/to/save', 'w').write(encoded_img.eval()) 将解码后的图像存储起来
'''
with tf.gfile.FastGFile('','r') as f:
	undecode_img = f.read()
	decode_img = tf.image.decode_jpeg(undecode_img)
with tf.gfile.FastGFile('','w') as f:
	f.write(decode_img.eval())

'''
2、图像大小调整：
   保证信息尽量少的丢失：tf.image.resize_images(image_data, [300,300], method)
      首先将图片数据类型转换为实数类型，会将0-255的像素值转为0.0-1.0的实数
      可以避免有些API不支持整数的问题，也可以减少数值类型转换过程中的精度损失
   通过剪裁或者填充的方式来调整图像大小：tf.image.resize_image_with_crop_or_pad(img_data, 300, 1000)
      如果图像尺寸大于目标尺寸，会自动截取图像中居中的部分
      如果图像尺寸小于目标尺寸，会自动在原始图像四周填充0背景
   按比例剪裁图像大小：tf.image.central_crop(image_data,rate)
      rate：0-1的实数，居中剪裁图像
'''

'''
3、图像翻转
   图像上下翻转：tf.image.flip_up_down(image_data)
   图像左右翻转：tf.image.flip_left_right(image_data)
   50%概率上下翻转图像：tf.image.random_flip_up_down(image_data)
   50%改路左右翻转图像：tf.image.random_flip_left_right(image_data)
'''

'''
4、亮度调整：
   图像亮度调整：tf.image.adjust_brightness(image_data, -0.5)
       可能会导致图像亮度超过0-1的范围，需要有截断处理。
       如果有多项图像处理操作，应该在所有操作完成后执行截断操作
   在[-max_delta, max_delta)范围内随机调整图像的亮度：tf.image.random_brightness(image, max_delta)
   图像亮度截断操作：tf.image.clip_by_value(image_data, 0.0, 1.0)
5、对比度调整：
   图像对比度调整：tf.image.adjust_constrast(image_data, 0.5)
   随机调整图像的对比度：tf.image.random_constrast(image_data, lower, upper)
6、色相调整
   图像色相调整：tf.image.adjust_hue(image_data, 0.1)
   随机图像色相调整：tf.image.random_adjust_hue(image_data, max_delta)
7、饱和度调整
   图像饱和度调整：tf.image.adjust_saturation(image_data, -5)
   随机饱和度调整：tf.image.random_adjust_saturation(image_data, lower, upper)
8、标准化调整
   图像标准化调整：tf.image.per_image_standardization(image_data)
'''

'''
9、标注框处理：
   在图像中加入标注框：tf.image.draw_bounding_boxes(batched, boxes)
     该函数要求图像矩阵中的数字是实数类型，首先需要将图像转为实数类型tf.image.covert_image_dtype()
     batched：函数需要的是一个batch的图像，需要做维度填充：tf.image.expand_dims(image_data, 0)
     boxes:表示每个图像的所有标注框，[[[ymin, xmin, ymax, xmax],[ymin,xmin,ymax,xmax]]]
        使用的是相对位置，三层嵌套表示图像，图像中的多个框
   随机截取图像：tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_bosex=boxes, mnin_object_covered=0.4)
     返回begin，size，bbox_for_draw
     distored_image = tf.slice(image_data, begin,size)
'''

### 多线程处理输入数据框架
'''
为了避免数据预处理操作成为模型训练的性能瓶颈，TF提供了多线程处理输入数据的框架
基本处理流程：1、指定原始数据的文件列表
             2、创建文件列表队列
             3、从文件中读取数据
             4、数据预处理
             5、整理成Batch作为神经网络输入
'''

'''
在TF中，队列和变量类似，都是计算图上的有状态节点，其他节点可以修改它们的状态。可以通过赋值操作修改变量的状态
对于队列, 可以通过enqueue、enqueue_many、dequeue来修改队列状态
'''
# 创建一个先入先出的队列，队列中最多可以保存两个元素，并指定类型为整数
q = tf.FIFOQueue(2, 'int32')
# 创建一个随机队列，会将元素顺序打乱，每次出队列的操作得到的是当前队列所有元素中随机的一个
rq = tf.RandomShuffleQueue(2,'int32')
# 使用enqueue_many函数来初始化队列中的元素，在使用队列之前需要明确的调用初始化过程
init = q.enqueue_many(([1,2],))   # 要求参数是一个tensor列表
# 使用dequeue函数从队列中获取一个元素
a = q.dequeue()
a += 1 # 对获取的元素进行计算操作
q_inc = q.enqueue([a])  # 要求参数是一个tensor

with tf.Session() as sess:
	sess.run(init)
	for i in range(5):
		print(sess.run([a, q_inc]))

'''
TF提供了tf.train.Coordinator和tf.train.QueueRunner两个类来完成线程协同的功能。
   tf.Coordinator类主要用来协同多个线程一起停止，并提供了should_stop、request_stop和join三个函数。
   在线程启动之前，需要先声明一个tf.Coordinator类，并将这个类传入每一个创建的线程中。
   启动的线程需要一直查询tf.Coordinator类中提供的should_stop函数，当这个函数返回值为True时，当前线程要退出
   每一个启动的线程可以通过调用request_stop函数来通知其他线程退出。
'''
import numpy as np
import threading
import time

# worker函数
def loop(coord, work_id):
	while not coord.should_stop():
		if np.random.rand() < 0.1:
			print('stopping from id{}'.format(work_id))
			coord.request_stop()
		else:
			print('working on id{}'.format(work_id))
		time.sleep(1)

# 新建coord类
coord = tf.train.Coordinator()
# 创建线程
threads = [threading.Thread(target=loop, args=(coord,_)) for _ in range(10)]
# 启动所有线程
for t in threads: t.start()
# 阻塞主进程，专注执行工作线程的东西，等待所有线程退出
coord.join(threads)

'''
tf.train.QueueRunner主要用于启动多个线程来操作同一个队列，启动的线程可以通过tf.Coordinator类来统一管理
'''
# 定义先进先出队列
queue = tf.FIFOQueue(100, 'int32')
# 定义入队操作
queue_op = queue.enqueue([tf.random_normal([1], dtype='int32')])
# 使用QueueRunner来创建多个线程运行队列的入队操作
# 第一个参数表示需要操作的队列，第二个参数表示启动5个线程来完成队列的入队操作
qr = tf.train.QueueRunner(queue, [queue_op] * 5)
# 将其加入计算图的指定集合中,没有指定集合，会加入QUEUE_RUNNERS默认集合中
tf.train.add_queue_runner(qr)
# 定义出队列操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
	# 定义coord来协同启动线程
	coord = tf.train.Coordinator()
	# 使用QueueRunner时，需要明确调用start_queue_runners来启动所有线程，该函数或默认启动QUEUE_RUNNERS集合中
	# 的所有queuerunner,所有add函数和start函数会指定同一个集合
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for _ in range(3): print(sess.run(out_tensor)[0])
	# 使用coord来停止线程
	coord.request_stop()
	# 阻塞主进程
	coord.join(threads)



#### 输入文件队列
'''
TF使用队列管理输入文件列表
使用tf.train.match_filenames_once函数来获取一个符合正则表达式的所有文件列表；
使用tf.train.string_input_producer函数对文件列表进行有效管理，
  该函数会使用初始化时提供的文件列表创建一个输入队列，输入队列中的元素为文件列表中的所有文件。
使用tf.TFRecordReader实例的read函数从输入队列中获取文件，
  每次调用文件读取函数时，该函数会先判断当前是否已有打开的文件可读，如果没有打开或者打开的已经读完，
  read函数会从输入队列中出队一个文件并从这个文件中读取数据
'''
# 根据正则表达式获取对应的文件列表
files = tf.train.match_filenames_once('path/to/data.tfrecord-*')
# 获取输入文件队列, shuffle参数可以保证文件在加入队列之前是乱序的，所以出队列的时候也是乱序的
# 随机打乱文件顺序以及加入输入队列的过程会跑在一个单独的线程，这样不会影响获取文件的速度
filequeue = tf.train.string_input_producer(files, shuffle=False)

# 输入文件读入接口
reader = tf.TFRecordReader()
# 从输入队列中获取文件
_, serialized_example = reader.read(filequeue)

features = tf.parse_single_example(serialized_example, features={
	'i': tf.FixedLenFeature([], tf.int64),
	'j': tf.FixedLenFeature([], tf.int64)
})

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(files))

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for i in range(6):
		print(sess.run([features['i'], features['j']]))
	coord.request_stop()
	coord.join(threads)

# 从文件列表中获得单个样本之后，采用图像预处理函数，然后组织成batch提供给网络的输入层
# TF提供了两个生成batch的函数：tf.train.batch/ tf.train.shuffle_batch
# 这两个函数都会生成一个队列，队列的入队操作是生成单个样例的方法，每次出队得到的是一个batch的样例
example, label = features['i'], features['j']
batch_size = 3
capacity = 1000 + 3 * batch_size

# 使用batch函数组合样例，[example, label]参数给出了需要组合的元素
# batch_size表示每个batch中元素的个数，capacity表示队列的最大容量
# 当队列长度等于容量时，TF会暂停入队操作，而只是等待元素出队
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

# min_after_dequeue参数是shuffle_batch函数特有的，限制了出队时队列中元素的最少个数
# 当队列中元素个数太少，随机打乱样例顺序的作用不太大，所以会等入队操作后在输出
random_example_batch, random_label_batch = tf.train.shuffle_batch([example_batch, label_batch],batch_size=batch_size,
                                                                  capacity=capacity, min_after_dequeue=30)

# tf.train.shuffle_batch函数和tf.train.shuffle_batch_join函数可以完成多线程并行的方式进行数据预处理
# 根据设定参数num_threads来设置线程数
# shuffle_batch函数读取同一个文件
# shuffle_batch_join函数读取不同文件

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	for i in range(3):
		cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
		print(cur_example_batch, cur_label_batch)
	coord.request_stop()
	coord.join(threads)


#### TF的数据处理框架（Dataset）
'''
以上介绍了通过队列进行多行程输入的方法，TF提供了一个更高层的数据处理框架
在新框架中，每个数据来源被抽象成一个‘数据集’，开发者可以以数据集为基本对象，
方便的进行batching、shuffle等操作。
利用数据集读取数据有三个基本步骤：
   1、定义数据集的构造方法。
      从张量中构建数据集：tf.data.Dataset.from_tensor_slices()
      从文件中构建数据集：tf.data.TextLineDataset()
      从TFRecord中构建数据集：tf.data.TFRecordDataset(),因为TFRecord都有自己不同的feature格式，
         需要提供一个parse函数来解析读取到的TFRecord数据格式。
         dataset = tf.data.TFRecordDataset().map(parse): map函数表示对数据集中每一条数据进行条用相应的解析方法
   2、定义遍历器。
      make_one_shot_iterator：要求数据集的所有参数已经确定，因此不需要特别的初始化过程
      make_initializable_iterator：使用了placeholder来初始化数据集，动态初始化数据集的
                                   可以不用将数据集路径参数写入计算图中，使用程序定义的方式传入
   3、使用get_next()方法从遍历器中读取数据张量，作为计算图其他部分的输入
'''

import tensorflow as tf

def parser(record):
	features = tf.parse_single_example(record,features={
		'img': tf.FixedLenFeature([], tf.int64),
		'label': tf.FixedLenFeature([], tf.int64)
	})
	return features['img'], features['label']
input_files = ['tfrecord1', 'tfrecord2']

dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)

# map方法返回的是一个新的数据集，可以继续调用高层API

# buffer_size参数相当于min_after_dequeue，shuffle算法在内部使用一个缓冲区保存buffer_size条数据
# 需要读入数据时，从缓冲区中随机选择一条输出，size越大随机效果越好，但是占用内存越大
dataset.shuffle(buffer_size=100)
# 将数据按batch输出，如果数据集中包含多个张量，batch操作将对每个张量分开进行（image和label分开）
dataset.batch(batch_size=64)
# 将数据集中的数据复制多份，其中每一份称为epoch。
# 如果在repeat前进行shuffle，输出的每个epoch的shuffle结果是不同的。也是一个计算节点
dataset.repeat(10)
## concatenate()/take()/skip()/flap_map()

iterator = dataset.make_one_shot_iterator()
img, label = iterator.get_next()

input_files_placeholder = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files_placeholder)

dataset = dataset.map(parser)
n_iterator = dataset.make_initializable_iterator()
img, label = n_iterator.get_next()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print(sess.run([img, label], feed_dict={input_files_placeholder:['path/to/tfrecord1','path/to/tfrecord2']}))