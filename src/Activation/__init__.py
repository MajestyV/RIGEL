# This package aims to provide different kind of activation function used for the implementation of pure algorithmic
# echo state network and its derivatives. More importantly, of course, simulating the nonlinear activation functions
# provided by electronic, optical and opto-electronic device and hardware systems.

# 标准激活函数
from .Standard import Linear, ReLU, dReLU  # 线性以及有限非线性激活函数
from .Standard import Tanh, Tanh_flipped, Sigmoid, Sigmoid_flipped  # 挤压型激活函数

# 硬件激活函数
from .NonlinearNode import I_Taylor
from .HardwareReservoirNeuron import ReservoirNeuron_SourceFollower, ReservoirNeuron_SourceFollower_central
from .HardwareReservoirNeuron import ReservoirNeuron_SourceFollower_matrix
from .HardwareReservoirNeuron import ReservoirNeuron_SourceFollower_central_matrix
# 由GPRA计算得到的唯象物理模型
from .HardwareReservoirNeuron import SourceFollower_DevVer
from .HardwareReservoirNeuron import SourceFollower_HIT