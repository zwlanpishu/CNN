import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weight = tf.get_variable(name = "Weight", shape = (), dtype = tf.float32, 
                         initializer = tf.random_uniform_initializer())
bias = tf.get_variable(name = "bias", shape = (), dtype = tf.float32,
                       initializer = tf.zeros_initializer())

y = Weight * x_data + bias

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(201) : 
    _, temp_loss = sess.run([optimizer, loss])
    if i % 20 == 0 : 
        print(i, temp_loss, sess.run(Weight), sess.run(bias))

### create tensorflow structure end ###