# 此函数包专门用于数据的可视化及保存，故命名为Gallery（画廊）

# 导入色彩模块
from . import colors as colors
from .colors.ColorConvertion import RGB, CMYK_to_RGB  # 导入色值转换函数（由于函数数目较少，采用直接调用方式，这样在外部调用时便无需声明具体文件）
# 以下是一些以字典形式储存的色值
from .colors.Seaborn_xkcd import xkcd_rgb
from .colors.Seaborn_crayons import crayons

# 画图初始化设置模块
from .Initialize import GlobalSetting   # 画图初始化设置函数

# 基础画图模块
from . import Visualization_basic as Visualization_basic

# 动态系统可视化模块
from .Visual_DynamicalSystems import QuickView_1D  # 一维动态系统
from .Visual_DynamicalSystems import QuickView_3D_TimeSequence,QuickView_3D_Trajectory  # 三维动态系统
from .Visual_DynamicalSystems import QuickView_Spatiotemporal  # 时空混沌序列

# 网络拟合性能分析模块
from .Visual import Analyze_1D_systems  # 一维动态系统
from .Visual import Analyze_3D_systems, Analyze_3D_systems_synchronization  # 三维动态系统
from .Visual import Analyze_Spatiotemporal  # 时空混沌序列