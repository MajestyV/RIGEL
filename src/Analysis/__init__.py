# This module aim to provide in-depth analysis of the target systems and the corresponding network.

# Lyapunov指数分析模块，使用于动态系统回归任务
from .LyapunovExponentAnalysis import MaximumLyapunovExponent  # 此函数可以计算最大Lyapunov指数（MLE）
from .LyapunovExponentAnalysis import Quick_LyapunovAnalysis   # 此函数可对目标系统以及算法模型的拟合结果进行快速李雅普诺夫指数分析

# RNN网络memory capacity分析模块，用于分析网络拓扑的记忆能力，以及衡量memory-nonlinearity trade-off
from .MemoryCapacity import Get_MemoryCapacity_Dataset  # 此函数可以从Dataset文件夹导入预先准备好的数据集
from .MemoryCapacity import MemoryCapacity  # 此函数可以计算分析RNN的memory capacity