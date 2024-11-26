# 导入各种误差评估函数，用于判别算法模型的额性能，方便后续调用
# 神经网络或者说机器学习算法的主要研究问题或者说主要任务可以分为两大类：I. 分类（Classification）；II. 回归（Regression）
# 详情请参考：https://zhuanlan.zhihu.com/p/86120987

# 分类任务专用的衡量指标

# 回归任务专用的衡量指标
from .Regression import Standard_deviation, STD
from .Regression import Deviation, MSE, RMSE, NRMSE
# 一些特殊的衡量指标
from .Regression import NRMSE_ICCAD
