import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


"""
pad函数
输入：X 输入的所有样本数据
      pad 进行边缘填充时，每个维度填充的像素个数
输出：X_pad 边缘填充之后的所有样本数据 
功能：对输入的图像样本数据进行边缘填充
"""
def zero_pad(X, pad) :

    # numpy有专门的边缘填充函数
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 
                   "constant", constant_values = 0)
    return X_pad

"""
单步卷积函数：
输入：a_slice_prev 卷积窗口对应的小数据立方体
      Weight 单次卷积算子
      bias 单次卷积的偏差
输出：Z 单次卷积运算的结果
功能：用于进行原数据立方体中，每次卷积窗口对应的小立方体与卷积算子之间的线性运算
"""
def conv_single_step(a_slice_prev, Weight, bias) :
    
    # 小立方体与卷积算子，二者必须是相同的形状维度
    assert(a_slice_prev.shape == Weight.shape)

    # 最后输出的是一个标量数据
    Z = np.squeeze(np.sum(np.multiply(a_slice_prev, Weight)) + bias)

    return Z

"""
前向卷积函数：
输入：A_prev 输入的训练样本
      Weight 卷积算子
      bias 卷积偏差
      hparameters 系统的超参数
输出：输入样本的卷积运算结果
功能：针对所有输入的训练样本的一次完整卷积运算
"""
def conv_forward(A_prev, Weight, bias, hparameters) : 

    # 获取所有训练样本的各个维度信息，其中num为样本标号，其余为三个维度信息
    (num, height_prev, width_prev, channels_prev_A) = A_prev.shape

    # 获取卷积算子的各个维度信息，其中channels为对应的卷积算子的个数
    (f_size_h, f_size_w, channels_prev_W, channels) = Weight.shape

    # 单个卷积算子的通道数与样本通道数必须一致
    assert(channels_prev_A == channels_prev_W)

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 卷积输出立方体的高、宽维度计算，通道数为卷积算子的个数
    height = int((height_prev + 2 * pad - f_size_h) / (stride)) + 1
    width = int((width_prev + 2 * pad - f_size_w) / (stride)) + 1

    # 卷积输出立方体初始化
    Z = np.zeros((num, height, width, channels))
    assert(bias.shape == Z.shape)
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range (num) :

        # 取出某个特定样本进行前向卷积，python中对a_prev_pad操作时，也会改变A_prev_pad值
        a_prev_pad = A_prev_pad[i]
        for h in range (height) :
            for w in range (width) : 
                for c in range (channels) : 
                    
                    # 确定当前卷积窗口对应的立方体数据
                    vert_start = h * stride
                    vert_end = h * stride + f_size_h
                    hori_start = w * stride
                    hori_end = w * stride + f_size_w
                    a_prev_pad_slice = a_prev_pad[vert_start : vert_end, 
                                                  hori_start : hori_end, :]
                    
                    # 进行单步卷积，并将单步卷积的结果存入到输出的数据立方体中
                    Z[i, h, w, c] = conv_single_step(a_prev_pad_slice, 
                                                     Weight[:, :, :, c], 
                                                     bias[i, h, w, c])

    # 判断输出数据的维度是否正确
    assert(Z.shape == (num, height, width, channels))

    # 卷积完成后将对应的数据存入缓存，供后续反向传播时调用
    cache = (A_prev, Weight, bias, hparameters)
    return Z, cache

"""
前向池化函数：
输入：A_prev 所有的训练样本
      hparameters 系统的超参数
      mode 池化模式
输出：输入训练样本的池化运算的结果
功能：针对所有输入的训练样本进行池化
"""
def pool_forward(A_prev, hparameters, mode = "max") :

    # 获取所有训练样本的各个维度信息，其中num为样本标号，其余为三个维度信息
    (num, height_prev, width_prev, channels_prev) = A_prev.shape

    # 获取池化算子的维度信息，池化算子不同于卷积算子，针对一个样本只用一次
    stride = hparameters["stride"]
    f = hparameters["f"]

    # 池化输出立方体的高、宽维度计算，通道数为原本对应样本的通道数
    height = int((height_prev - f) / (stride)) + 1
    width = int((width_prev - f) / (stride)) + 1
    A = np.zeros((num, height, width, channels_prev))

    for i in range (num) :

        # 取出某个特定样本进行前向池化
        a_prev = A_prev[i]
        for h in range (height) : 
            for w in range (width) : 
                for c in range (channels_prev) : 
                    
                    # 确定当前池化窗口对应的立方体数据
                    vert_start = h * stride
                    vert_end = h * stride + f
                    hori_start = w * stride
                    hori_end = w * stride + f
                    a_prev_slice = a_prev[vert_start : vert_end, 
                                          hori_start : hori_end, :]

                    # 获得当前的池化输出
                    if (mode == "max") :
                        A[i, h, w, c] = np.max(a_prev_slice[:, :, c])
                    elif (mode == "average") : 
                        A[i, h, w, c] = np.mean(a_prev_slice[:, :, c])
    
    # 判断输出数据的维度是否正确
    assert(A.shape == (num, height, width, channels_prev))

    # 池化完成后将对应的数据存入缓存，供后续反向传播时调用
    cache = (A_prev, hparameters)
    return A, cache

"""
卷积层反向传播：
输入：dZ 损失函数相对于卷积层每个输出的导数
      cache 系统缓存，用于获取计算反向传播需要的数据
输出：损失函数针对卷积层所有输入的导数
功能：进行卷积层的反向传播计算
"""
def conv_backward(dZ, cache) : 

    # 从系统缓存中获取计算反向传播需要的数据
    (A_prev, Weight, bias, hparameters) = cache

    # 获取对应的卷积层输入数据维度信息
    (num, height_prev, width_prev, channels_prev) = A_prev.shape

    # 获取卷积算子对应的维度信息
    (f_size_h, f_size_w, channels_prev, channels) = Weight.shape

    # 获取导数输入的维度信息
    (num, height, width, channels) = dZ.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 初始化保存反向传播结果的数据的维度信息
    dA_prev = np.zeros((num, height_prev, width_prev, channels_prev))
    dW = np.zeros((f_size_h, f_size_w, channels_prev, channels))
    db = np.zeros((num, height, width, channels))

    # 进行PAD处理
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range (num) : 

        # 针对某个特定标号的训练样本进行反向传播
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        # 针对某个特定标号样本对应的dZ中的某个数据，通过链式法则求其对应的da_prev_pad
        for h in range (height) : 
            for w in range (width) : 
                for c in range (channels) :  
                    
                    # 确定当前卷积窗口对应的小立方体数据
                    vert_start = h * stride
                    vert_end = vert_start + f_size_h
                    hori_start = w * stride
                    hori_end = hori_start + f_size_w
                    a_slice = a_prev_pad[vert_start : vert_end, hori_start : hori_end, :]

                    # 更新当前卷积窗口对应的小立方体中的各个输入变量的导数
                    # 两个更新公式与经典神经网络的更新推导过程一样
                    # 需要注意的是要使用累加运算和点乘运算
                    da_prev_pad[vert_start : vert_end, hori_start : hori_end, :] += (
                        Weight[:, :, :, c] * dZ[i, h, w, c])
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
    
        # db的值与dZ的值始终一样
        db[i, :, :, :] = dZ[i, :, :, :]

        # 从pad之后的dA_prev_pad中只取出对应的针对原始dA_prev的部分
        # 其余的变量的导数不需要前向传播
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad : -pad, pad : -pad , :]

    # 判断输出数据的维度是否正确
    assert(dA_prev.shape == (num, height_prev, width_prev, channels_prev))
    return dA_prev, dW, db

"""
最大池化掩码生成：
输入：x nxn固定大小的窗口
输出：mask 返回该窗口对应的掩码，其中最大值元素对应的值为true，其余为false
功能：掩码生成，暂不考虑x有多个相同最大值情况
"""
def create_mask_from_window(x) : 
    mask = (np.max(x) == x)
    return mask

"""
标量矩阵化分布：
输入：dz 输入的标量信息
      shape 对应的要矩阵化的维度
输出：a 矩阵
功能：将一个输入标量转为输出矩阵
"""
def distribute_value(dz, shape) : 
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average
    return a

"""
池化层反向传播：
输入：dA 损失函数相对于所有池化层输出的导数
      cache 系统保存的用于计算反向传播的数据
      mode 池化模式
输出：dA_prev 损失函数相对于所有池化层输入的导数
功能：进行池化层的反向传播计算
"""
def pool_backward(dA, cache, mode = "max") : 
    
    # 获取池化层反向传播需要的数据
    (A_prev, hparameters) = cache

    # 获取池化算子对应的步长以及维度大小信息
    stride = hparameters["stride"]
    f = hparameters["f"]

    # 获取池化层输入的样本数据的维度信息
    (num, height_prev, width_prev, channels_prev) = A_prev.shape

    # 获取损失函数相对于池化层输出的导数的维度信息
    (num, height, width, channels) = dA.shape

    # 初始化损失函数相对于池化层输入的导数的数据结构
    dA_prev = np.zeros_like(A_prev)

    for i in range (num) : 
        
        # 针对池化层输入中某个特定的样本进行反向传播
        a_prev = A_prev[i]

        # 针对池化层输出中某个特定分量进行计算
        for h in range (height) : 
            for w in range (width) : 
                for c in range (channels) : 
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f 

                    if mode == "max" :

                        # 池化是基于通道的，这里的数据截取与卷积时有区别
                        a_prev_slice = a_prev[vert_start : vert_end, 
                                              horiz_start : horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += (
                            mask * dA[i, h, w, c])

                    if mode == "average" : 
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += (
                            distribute_value(da, shape))
                    
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev

"""
需要新增卷积层的活化函数以及卷积层活化函数对应的反向传播
"""