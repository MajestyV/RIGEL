# RIGEL - Reservoir computing Integrated with Generalized and Efficient Learning

# NOTE: 以下是RIGEL的初始化导入语句
import os, toml
current_path = os.path.abspath(os.path.dirname(__file__))  # 获取文件目录
project_path = current_path[:current_path.find('RIGEL') + len('RIGEL')]  # 获取项目根路径，内容为当前项目的名字，即RIGEL

configs = toml.load(f'{current_path}/configs.toml')  # 读取配置文件

default_results_dir = f'{project_path}/{configs['default_dir_name']['results']}'  # 默认结果目录
default_demo_dir = f'{project_path}/{configs['default_dir_name']['demo']}'        # 默认演示目录
default_gallery_dir = f'{project_path}/{configs['default_dir_name']['gallery']}'  # 默认图库目录

# NOTE: 以下是RIGEL各种模块的导入语句
from . import utils                                                        # 工具函数包

# 硬件分析模块
from . import AnalysisToolkit as AnalysisToolkit                           # 硬件分析工具包

# 数据集生成及整理模块
from . import DynamicalSystemGenerator as Dynamics                         # 动态系统数据集模块

# Dataset_makeup()函数可以将时序信号整理为Reservoir computing所需的数据集
from .OrganizingDataset import Dataset_makeup, Dataset_Makeup4ERA
from .OrganizingDataset import Gen_NARMA                                   # 此函数可以快速生成NARMA数据集

# 权重生成模块
from .Weight_Generation import RandomWeightMatrix, NormalizeMatrixElement  # 随机矩阵生成函数
from .Weight_Generation import Network_initial                             # 导入通过networkx生成稀疏连接权重的函数

# 权重优化模块
from .Weight_Optimization import WeightOptimizing

# 激活函数模块
from . import Activation as Activation
from .Activation import Standard                                           # 标准挤压型激活函数（tanh、sigmoid等）
from .Activation import NonlinearNode                                      # 器件激活函数-多项式回归拟合

# 算法模型
from . import DelayRC as DelayRC
from . import EchoStateNetworks as ESN

# 模型评估模块
from .Evaluation import Regression                                         # 用于评估回归任务表现的函数包
from .Evaluation import TrajectorySimilarity                               # 用于评估轨迹相似性的函数包

# 数据可视化及保存模块
from . import VISION as VISION
from .VISION import colors                                                  # 导入色彩模块
from .VISION import Visualization                                           # 可视化函数包
from .VISION import DataRecording                                           # 数据保存函数包
from .VISION import QuickView                                               # 快速可视化函数包

# 接口模块
from . import API_for_DataEncryption as API_for_DataEncryption              # 数据加密功能接口
