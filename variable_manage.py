import tensorflow as tf

# TF中，变量(tf.Variable)的作用就是保存和更新神经网络中的参数
# 声明变量时，给出对应的初始值
weights = tf.Variable(tf.random_normal([2,3], stddev=0.2, name='w'))
# 其中，初始值一般使用随机函数生成

# TF中的随机数生成函数如下：
'''
tf.random_normal: 正太分布， 主要参数：平均值、标准差、取值类型
tf.truncated_normal: 正太分布(随机出来的值超过两个标准差，重新选取)，主要参数：平均值、标准差、取值类型
tf.random_uniform: 均匀分布，主要参数：最小、最大值、取值类型
tf.random_gamma: Gamma分布，主要参数：形状参数alpha、尺度参数bata、取值类型
'''
bias = tf.Variable(tf.ones([10]))
# TF中的常量生成函数如下：
'''
tf.zeros: 产生全0数组， tf.zeros([2,3], int32)
tf.ones: 产生全1数组，tf.ones([2,3], int32)
tf.fill: 产生一个全部为给定数字的数组，tf.fill([2,3],9)
tf.constant: 产生一个给定值的常量，tf.constant([1,2,3])/tf.constant([[1,2],[2,3]])
'''
# 通过其他变量的初始值来初始化新的变量
# 一个变量的值在被正式使用之前，变量的初始化过程需要被明确的调用
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value() * 2.0)

# 张量和变量的关系
'''
TF的核心概念是张量，所有的数据都是通过张量的形式组织起来的；
在TF中，变量的声明函数tf.Variable是一个运算，运算的输出结果就是张量：变量只是一个特殊的张量
变量通过read操作将变量值（张量）提供给计算节点；
变量通过Assign操作将随机数生成函数的输出作为变量的输入，完成变量的初始化；
'''

# 变量的维度和类型
'''
类似于张量，变量的维度和类型是变量的两个重要属性。
变量的类型是不可变的，
维度通过设置（validate_shape=False），维度是可变的，但是很少这样做
'''
print(weights.dtype, weights.shape)
