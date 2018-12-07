import numpy as np
import tensorflow as tf

### 构建计算图 ###
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], 
                       [2]])

product = tf.matmul(matrix1, matrix2)

### session的第一种用法 ###
sess = tf.Session()
ans = sess.run(product)
sess.close()

print("ans: ", ans)

### session的第二种用法 ###
with tf.Session() as sess :
    res = sess.run(product)

print("res: ", res)
