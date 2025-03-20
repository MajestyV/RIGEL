import os
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator, RegularGridInterpolator  # scipy中比较现代化高效的用于高维散点插值的函数

# 画图工具包
import seaborn as sns
import matplotlib.pyplot as plt

# from src.Evaluation import Hausdorff_distance

if __name__ == '__main__':
    working_dir = 'E:/PhD_research/NonlinearNode/Simulation/RIGEL/Working_dir/AnalogESN_FSJ_clipped_NoiseAnalysis/HausdorffDistance'

    # file_list = os.listdir(working_dir)  # 获取文件夹中所有文件的名称（这个方法名称顺序是乱的，不适用）
    file_list = [f'AnalogESN_FSJ_clipped_HausdorffDistance_normal_SNR-{i}.txt' for i in range(10, 210, 10)]

    print(file_list)
    num_files = len(file_list)  # 获取文件夹中文件的数量

    Hausdorff_dist_map = np.empty((20, 30))  # 创建一个20*30的矩阵用于存放Hausdorff距离数据

    for i in range(num_files):
        filename = file_list[i]  # 获取文件的名称
        data_file_path = os.path.join(working_dir, filename)  # 获取文件的路径

        data = np.loadtxt(data_file_path)  # 读取数据

        # Hausdorff_dist_map[i] = data[:,1]  # 将数据存入矩阵中
        Hausdorff_dist_map[i] = np.log10(data[:,1])  # 将数据存入矩阵中（取对数）

        # print(data)  # 打印数据，查看数据用于debug

    # 画图模块
    npoints_eval = np.arange(100, 3100, 100)  # 衡量点数
    SNR = np.arange(10, 210, 10)  # 信噪比

    # xx0, yy0 = np.meshgrid(npoints_eval, SNR)

    # print(xx0)

    # Hausdorff_dist_func = RBFInterpolator((npoints_eval, SNR), Hausdorff_dist_map, kernel='gaussian')  # 创建插值函数
    Hausdorff_dist_func = RegularGridInterpolator((SNR, npoints_eval), Hausdorff_dist_map)  # 创建插值函数

    x1 = np.linspace(100, 3000, 300)
    y1 = np.linspace(10, 200, 300)
    xx1, yy1 = np.meshgrid(x1, y1)

    zz1 = Hausdorff_dist_func((yy1, xx1))  # 插值

    # zz = griddata((npoints_eval, SNR), Hausdorff_dist_map, (xx, yy), method='gaussian')  # 插值

    # print(zz1)


    # print(xx.shape)


    # print(npoints_eval)
    # print(SNR)

    # exit()


    # print(file_list)

    # data_file =

    # 画图参数
    # colormap = 'RdBu_r'  # 高级颜色
    colormap = 'coolwarm'

    # sns.heatmap(Hausdorff_dist_map, vmin=-2, vmax=1, cmap=colormap)
    sns.heatmap(zz1, vmin=-2, vmax=1, cmap=colormap)

    # 保存数据
    saving_dir = 'E:/PhD_research/NonlinearNode/Simulation/RIGEL/Working_dir'
    for fmt in ['eps', 'png', 'pdf']:
        plt.savefig(f'{saving_dir}/HausdorffDistance_Mapping.{fmt}', format=fmt)

    plt.show(block=False)