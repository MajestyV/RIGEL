# 此代码可以对利用matplotlib绘制的图像提供初始化设置

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 全局画图设置：一些用于文章级结果图的matplotlib参数，可以作为matplotlib的全局变量载入
def GlobalSetting(**kwargs):
    # 设置刻度线方向
    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内

    # 设置全局字体选项
    font_type = kwargs['font_type'] if 'font_type' in kwargs else 'Arial'  # 默认字体为Arial
    font_config = {'font.family': font_type, 'font.weight': 'normal', 'font.size': 12}  # font.family设定所有字体为font_type
    plt.rcParams.update(font_config)  # 但是对于希腊字母(e.g. α, β, γ等)跟各种数学符号之类的不适用, Latex语法如Γ会被判断为None
    return