# 以下是各种不同的标准激活函数（activation function）

import numpy as np

########################################################################################################################
# 线性以及有限非线性激活函数

# 线性激活函数
def Linear(x,k=1,b=0): return k*x+b  # 默认为identity

# ReLU - Rectified Linear Unit (线性整流函数)
def ReLU(x): return x * (x > 0)

def dReLU(x): return 1. * (x > 0)

########################################################################################################################
# 挤压型激活函数

# Hyperbolic tangent function
def Tanh(x): return np.tanh(x)

# 沿y轴镜面翻转的tanh（由于tanh关于原点中心对称，所以这也是中心翻转或者沿x轴翻转的tanh）
def Tanh_flipped(x): return -np.tanh(x)

# Sigmoid function
def Sigmoid(x): return 1.0/(1.0+np.exp(-x))

# 沿y轴镜面翻转的sigmoid
def Sigmoid_flipped(x): return 1.0-1.0/(1.0+np.exp(-(x-2)))

