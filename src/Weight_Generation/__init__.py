# 权重生成模块
from .WeightMatrix import RandomWeightMatrix, NormalizeMatrixElement  # 随机矩阵生成函数
from .WeightMatrix_Networkx import Network_initial  # 导入通过networkx生成稀疏连接权重的函数