import tensorflow as tf


print('constant op:----------------------------------')
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
	print('a:{}, b:{}'.format(sess.run(a), sess.run(b)))
	print('a+b={}'.format(sess.run(a+b)))
	print('a*b={}'.format(sess.run(a*b)))

print('variable op:----------------------------------')
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
	print('Addition with variable:{}'.format(sess.run(add, feed_dict={a:2, b:3})))
	print('Multiplication with variable:{}'.format(sess.run(mul, feed_dict={a:33, b:2})))
	print('a+b={}'.format(sess.run(a+b, feed_dict={a:12,b:12})))
	print('a*b={}'.format(sess.run(a*b, feed_dict={a:12,b:2})))


print('matrix op:----------------------------------')
m1 = tf.constant([[3,3]], dtype=tf.float32)  # 注意数据类型一致
m2 = tf.constant([[4,4]], dtype=tf.float32)
m3 = tf.constant([[2.],[2.]])

mat_add = m1 + m2
mat_mul = m1 * m3

with tf.Session() as sess:
	print('matrix add(m1+m2)={}'.format(sess.run(mat_add)))
	print('matrix multiplicaton(m1*m3)={}'.format(sess.run(mat_mul)))
	res = sess.run(mat_mul)
	print('use the tmp result name:{}'.format(res))