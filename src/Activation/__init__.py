# This package aims to provide different kind of activation function used for the implementation of pure algorithmic
# echo state network and its derivatives. More importantly, of course, simulating the nonlinear activation functions
# provided by electronic, optical and opto-electronic device and hardware systems.

# 标准激活函数
from .Standard import Linear, ReLU, dReLU  # 线性以及有限非线性激活函数
from .Standard import Tanh, Tanh_flipped, Sigmoid, Sigmoid_flipped  # 挤压型激活函数

# 硬件激活函数
from .NonlinearNode import I_Taylor, I_Taylor_w_deviation