import numpy as np
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

sess = tf.Session()
ans = sess.run(output, feed_dict = {input1 : 2, input2 : 5})
sess.close()

print(ans)

