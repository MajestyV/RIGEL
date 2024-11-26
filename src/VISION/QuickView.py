# 此代码专为快速可视化而设计
import matplotlib.pyplot as plt
from .colors import crayons
from .Visualization import GlobalSetting

# 此函数可以快速可视化2D轨迹
# 用于快速可视化的函数
def Plot2D(trajectory, color=crayons['Navy Blue'], mode='curve'):
    GlobalSetting()  # 设置全局变量
    if mode == 'curve':
        plt.plot(trajectory[0,:], trajectory[1,:], color=color)
    elif mode == 'scatter':
        plt.scatter(trajectory[0, :], trajectory[1, :], s=2.0, c=color)
    else:
        print('Please specify a valid mode ! ! !')
        exit()  # 结束代码运行
    plt.show(block=True)  # 显示图片（用console运行代码时，只有加这一句代码才能plot图）
    return

# 此函数可以快速可视化3D轨迹
# 用于快速可视化的函数
def Plot3D(trajectory, color=crayons['Navy Blue'], show_grid=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")  # 设置画布为三维图图像
    ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], color=color)  # 画图
    ax.set_xlabel("$x$")  # 更改坐标轴标签
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.grid(show_grid)  # 隐藏背景网格线
    plt.tight_layout()  # 防止画图时，图像分布失衡，部分文字显示被遮挡的情况
    plt.show(block=True)  # 显示图片（非console运行代码时，只有加这一句代码才能plot图）
    return