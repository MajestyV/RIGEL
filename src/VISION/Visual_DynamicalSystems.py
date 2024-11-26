import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from .colors import crayons  # 导入色彩值字典

########################################################################################################################
# Quick view系列：用以快速绘制动态系统图像，辅助分析
def QuickView_1D():
    return

def QuickView_3D_TimeSequence(time,dynamical_system):
    return

# 画三维
def QuickView_3D_Trajectory(dynamical_system,color=crayons['Navy Blue']):
    x,y,z = dynamical_system
    plt.plot(x, y, z, color=color, projection='3d')  # 绘制三维图图像
    plt.grid(False)                                  # 隐藏背景网格线
    plt.tight_layout()                               # 防止画图时，图像分布失衡，部分文字显示被遮挡的情况
    plt.show(block=True)                             # 显示图片（用console运行代码时，只有加这一句代码才能plot图）
    return

def QuickView_Spatiotemporal(dynamical_system,figsize=(8,2),levels=np.arange(-3.5, 3.5, 0.2),**kwargs):
    L, t = dynamical_system.shape  # 从动态系统数据中读取系统信息

    time = kwargs['time'] if 'time' in kwargs else np.linspace(start=0, stop=t, num=t)
    length = kwargs['length'] if 'length' in kwargs else np.linspace(start=0, stop=L, num=L)

    # plot the result
    fig, ax = plt.subplots(figsize=figsize)
    xx, tt = np.meshgrid(time, length)
    cs = ax.contourf(xx, tt, dynamical_system, levels=levels, cmap=cm.jet)
    fig.colorbar(cs)

    ax.set_xlabel("t")
    ax.set_ylabel("x")

    plt.tight_layout()  # 防止画图时，图像分布失衡，部分文字显示被遮挡的情况
    plt.show(block=True)
    return

if __name__=='__main__':
    pass