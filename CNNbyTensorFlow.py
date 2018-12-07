import math
import h5py
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

"""
调整输入数据的格式
载入的文件中训练样本X数据格式为n x h x w x c
训练样本对应的属性Y数据格式为1 x n
"""
X_train = X_train_orig / 255                          # 对所有的像素值做类归一处理
X_test = X_test_orig / 255
Y_train = convert_to_one_hot(Y_train_orig, 6).T       # 将单输出转为one-hot编码, (6 x n).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape))
print("number of test examples = " + str(X_test.shape))
print("Y_train shape" + str(Y_train.shape))
print("Y_test shape" + str(Y_test.shape))
conv_layers = {}

"""
创建关于输入输出数据的张量表示
"""
def create_placeholders(n_H0, n_W0, n_C0, n_y) : 
    X = tf.placeholder(name = "X", shape = (None, n_H0, n_W0, n_C0), dtype = tf.float32)
    Y = tf.placeholder(name = "Y", shape = (None, n_y), dtype = tf.float32)
    return X, Y

"""
初始化卷积核参数, 并返回该参数，这里只使用了两个卷积层
"""
def initialize_parameters() :
    tf.set_random_seed(1)

    # 这里使用两个卷积层，因此只需要初始化两个卷积核的参数
    Weight1 = tf.get_variable(name = "Weight1", shape = (4, 4, 3, 8), dtype = tf.float32, 
                              initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    Weight2 = tf.get_variable(name = "Weight2", shape = (2, 2, 8, 16), dtype = tf.float32,
                              initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"Weight1" : Weight1,
                  "Weight2" : Weight2}
    
    return parameters

"""
前向传播
网络结构：conv1-->relu-->pool-->conv2-->relu-->pool-->fc
"""
def forward_propagation(X, parameters) : 
    Weight1 = parameters["Weight1"]
    Weight2 = parameters["Weight2"]

    # conv1-->relu-->pool
    Z1 = tf.nn.conv2d(input = X, filter = Weight1, 
                      strides = (1, 1, 1, 1), padding = "SAME")
    A1 = tf.nn.relu(Z1)
    A1_pool = tf.nn.max_pool(value = A1, ksize = (1, 8, 8, 1), 
                             strides = (1, 8, 8, 1), padding = "SAME")
    
    # conv2-->relu-->pool
    Z2 = tf.nn.conv2d(input = A1_pool, filter = Weight2,
                      strides = (1, 1, 1, 1), padding = "SAME")
    A2 = tf.nn.relu(Z2)
    A2_pool = tf.nn.max_pool(value = A2, ksize = (1, 4, 4, 1),
                             strides = (1, 4, 4, 1), padding = "SAME")

    # flatten to 1d vector
    A2_pool = tf.contrib.layers.flatten(inputs = A2_pool)

    # Apply a fully connected layer without an non-linear activation function. 
    # Do not call the softmax here. This will result in 6 neurons in the output layer, 
    # which then get passed later to a softmax. In TensorFlow, the softmax and cost 
    # function are lumped together into a single function, which you'll call in a 
    # different function when computing the cost. 
    Z3 = tf.contrib.layers.fully_connected(A2_pool, 6, activation_fn = None)
    return Z3

"""
计算前向传播后的损失值
"""
def compute_cost(Z3, Y) : 
    cost_all = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    cost = tf.reduce_mean(cost_all)
    return cost

"""
新增一张自己拍的图进行测试
"""
def read_picture() :
    fname_01 = "images/thumbs_02.jpg"
    image_01 = np.array(ndimage.imread(fname_01, flatten = False))
    my_image_01 = scipy.misc.imresize(image_01, size = (64,64))
    X_new = np.zeros((1, 64, 64, 3))
    X_new[0] = my_image_01
    return X_new
"""

构建卷积网络模型
"""
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 200, minibatch_size = 64, print_cost = True) : 
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (num, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    num_minibatches = int(num / minibatch_size)
    costs = []
    
    # forward propogation
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    
    # backward propagaiton
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # evaluate tensors
    init = tf.global_variables_initializer()

    with tf.Session() as sess : 
        sess.run(init)

        # evaluate the cost and optimizer
        for epoch in range(num_epochs) : 
            minibatch_cost = 0
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches : 
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X : minibatch_X, Y : minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            
            if (print_cost == True) and (epoch % 5 == 0) : 
                print("After epoch %d %f" %(epoch, minibatch_cost))
            if (print_cost == True) and (epoch % 1 == 0) : 
                costs.append(minibatch_cost)

        plt.plot(costs)
        plt.ylabel("cost")
        plt.xlabel("epoches")
        plt.title("learning rate = " + str(learning_rate))
        plt.show()

        # evaluate the accuracy
        predict = tf.arg_max(Z3, 1)
        reality = tf.arg_max(Y, 1)
        correct_prediction = tf.equal(predict, reality)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X : X_train, Y : Y_train})
        test_accuracy = accuracy.eval({X : X_test, Y : Y_test})
        
        print("Train Accuracy : " + str(train_accuracy))
        print("Test Accuracy ：" + str(test_accuracy))
        
        # predict the new picture
        X_new = read_picture()
        X_new_predict = sess.run(Z3, feed_dict = {X : X_new})
        print(X_new_predict)
    
        return train_accuracy, test_accuracy, parameters


"""
调用卷积网络模型，分别输出对训练样本和测试样本的预测准确度
"""
train_accuracy, test_accuracy, parameters = model(X_train, Y_train, X_test, Y_test)
