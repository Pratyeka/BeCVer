import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
# 为了避免在计算中出现类型不匹配的问题，建议对a、b变量指定dtype
result = a + b
# 张量在TF中并不是真正的数组，是对计算结果的引用
# 返回一个张量结构，属性：名字，维度，类型
# 张量用途：
# 1、对中间计算结果的引用，如a/b
# 2、获取计算结果，如:sess.run(result)
print(result)


# tf不会自动生成会话，需要手动生成会话
sess = tf.Session()
# 以下几种写法结果一致
with sess.as_default():
	print(result.eval())

print(sess.run(result))

print(result.eval(sess=sess))

# 非上下文环境需要手动关闭会话
sess.close()

# 通过ConfigProto protocol Buffer配置需要生成的会话
# 使用ConfigProto可以配置并行的线程数、GPU分配策略、运算超时时间等参数；
# 最常用的两个：allow_soft_placement和log_device_placement
"""
allow_soft_placement是bool类型值，为True时，当下面任一条件成立时，GPU上的计算
可以放到CPU上进行，
	1、运算无法在GPU上运行
	2、没有GPU资源
	3、运算输入包括对CPU计算结果的引用
log_device_placement是bool类型值，为True时，日志中将会记录每个节点被安排在哪个
设备上以方便调试，生产环境中参数设置为False来减少日志量
"""
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
sess1 = tf.Session(config=config)
