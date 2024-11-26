# 此函数可以利用matplotlib自定义创建色谱（colormap），用于热度图等，需要连续变化色值的场景

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import Normalize,ListedColormap, LinearSegmentedColormap

##########################################################
# Abbreviation:                                          #
# cmap - colormap，色谱                                   #
# nbins - number of bins，指定了颜色节点间插颜色值的数目       #
##########################################################

# 自定义色谱 - 创战记
Tron = LinearSegmentedColormap.from_list('Tron',
                                  [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0), (0.176, 0.976, 0.529), (0, 1, 1)])

# 自定义色谱 - 蓝调
Blues = ListedColormap(sns.color_palette('Blues').as_hex())

iColormap = {'Tron': Tron, 'Blues': Blues}