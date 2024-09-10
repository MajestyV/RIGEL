import numpy as np

# 此函数可以将保存在txt文件中的2D数组重构为3D数组
def Reform(data_deform):
    data_length, data_width = data_deform.shape  # 获取保存好的数据维度

    time = np.zeros((data_length,1))  # 保存时序信号的数组
    dynamical_system = np.zeros((data_length,data_width-1))  # 保存动态系统的数组

    for i in range(data_length):
        time[i,0] = data_deform[i,0]
        for j in range(1,data_width):
            dynamical_system[i,j-1] = data_deform[i,j]

    return time, dynamical_system

# 动态系统数据读取函数
def LoadDataset(datafile):
    data = np.loadtxt(datafile)
    return Reform(data)