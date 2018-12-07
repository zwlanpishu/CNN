import numpy as np
import tensorflow as tf

### 构建计算图 ###
state = tf.Variable(0, name = "state")
one = tf.constant(1, name = "one")

new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(0, 3) : 
    print(sess.run(update))
    print(sess.run(state))


