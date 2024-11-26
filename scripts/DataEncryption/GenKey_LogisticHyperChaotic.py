# 对于循环次数多的情况，在terminal运行会比IDE更快
# 有时pycharm的文件结构和cmd的文件结构不一样，在cmd中运行会显示：ModuleNotFoundError: No module named 'src'
# 这可以通过在脚本开头添加项目根目录到sys.path中解决，详情请参考：https://blog.csdn.net/qq_42730750/article/details/119799157
import os
import sys

script_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
project_path = os.path.abspath(os.path.join(script_path, '..'))  # 获取项目路径
sys.path.append(project_path)  # 添加路径到系统路径中

import numpy as np
from src import dynamics, Dataset_makeup, Activation, ESN, VISION, colors
import matplotlib.pyplot as plt

working_loc = 'default'  # 用于指定工作目录

saving_dir_dict = {'default': f"{project_path}/demo",
                   'Lingjiang': 'D:/Projects/NonlinearNode/Data/DataEncryption'}

# 用于快速可视化的函数
def QuickView(ground_truth, network_output,
              color_truth=colors.crayons['Navy Blue'], color_network=colors.crayons['Red Orange'], size=1.0):
    y_train, y_test = ground_truth  # 解压数据
    y_train_ESN, y_test_ESN = network_output

    # 设置画布
    VISION.GlobalSetting()  # 导入全局设置
    # gridspec_kw = {'wspace': 0.4, 'hspace': 0.4}  # 网格空间分配参数
    # fig, subplot_list = plt.subplots(nrows=2,ncols=2,gridspec_kw=gridspec_kw,figsize=(8,12))
    fig = plt.figure(figsize=(10, 4))  # 控制图像大小
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 空间分配，wspace和hspace可以调整子图间距

    training_phase = fig.add_subplot(1, 2, 1)
    testing_phase = fig.add_subplot(1, 2, 2)

    # 目标系统
    training_phase.scatter(y_test[0, :], y_test[1, :], c=color_truth, s=size)
    # 网络预测
    testing_phase.scatter(y_test_ESN[0, :], y_test_ESN[1, :], c=color_network, s=size)

    # 训练集
    #training_phase.scatter(y_train[0, :], y_train[1, :], c=color_truth, s=size)
    #training_phase.scatter(y_train_ESN[0, :], y_train_ESN[1, :], c=color_network, s=size)
    # 测试集
    #testing_phase.scatter(y_test[0, :], y_test[1, :], c=color_truth, s=size)
    #testing_phase.scatter(y_test_ESN[0, :], y_test_ESN[1, :], c=color_network, s=size)

    # 细节设置
    training_phase.set_xlim(-0.2,1.2)
    training_phase.set_ylim(-0.2,1.2)
    testing_phase.set_xlim(-0.2,1.2)
    testing_phase.set_ylim(-0.2,1.2)

    plt.show(block=True)

    return

if __name__=='__main__':
    num_step = 20001  # 总步数
    step_length = 1.0  # 步长

    time, data = dynamics.Logistic_hyperchaotic(num_step=num_step)

    num_init = 1000  # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
    num_train = 3000  # 训练集长度
    num_test = 5000  # 预测集长度
    initialization_set, training_set, testing_set = Dataset_makeup(time, data, num_init, num_train, num_test)

    # 解压数据
    t_init, x_init, y_init = initialization_set
    t_train, x_train, y_train = training_set
    t_test, x_test, y_test = testing_set

    ####################################################################################################################
    # 定义ESN网络
    # 先定义一些常用的网络参数
    input_scaling = 1.0  # 如果计算结果出现nan，则可以考虑先降低输入的缩放因子，因为我们的激活函数是无界函数，很容易超出计算机所能处理的量程
    # 水库权重矩阵的参数
    reservoir_dim = 300  # N是水库矩库的边长，同时也就是水库态向量的长度
    spectral_radius = 0.2
    reservoir_density = 0.2
    # 器件的性能的多项式拟合系数
    # device_coefficient = [0, -0.0606270243, 0.00364103237, 0.140685043, 0.00988703156, -0.00824646444,
    # -0.000618645284, 0.000257831028, 0.000011526794, -0.00000315380367]
    # reference_factor = 0.65
    transient = 0

    model = ESN.Analog_ESN(input_dimension=2,output_dimension=2,
                        input_scaling=input_scaling,
                        activation=Activation.I_Taylor,
                        reservoir_dimension=reservoir_dim,
                        reservoir_density=reservoir_density,
                        reservoir_spectral_radius=spectral_radius,
                        # reservoir_connection_weight=W_res,
                        transient=transient,
                        bias=0)

    # opt_algorithm=4的SelectKBest算法有奇效，太过夸张，慎用！！！主要是岭回归（opt_algorithm=2）效果太好！！！
    y_train_ESN, y_train, u_state_train, r_state_train, W_out = model.Training_phase(x_train, y_train, opt_algorithm=2)
    # 此模型可以利用transient参数先把前面一段储层的初始态去掉
    t_train_new = np.array([i + num_init + transient for i in range(y_train.shape[1])])

    y_test_ESN, u_state_test, r_state_test = model.Predicting_phase(num_test)

    # 可视化分析模块
    QuickView((y_train, y_test), (y_train_ESN, y_test_ESN))

    # 产生伪随机数串
    bias = 0.43715
    x, y = (y_test_ESN[0,:],y_test_ESN[1,:])
    bit_train = ''
    for i in range(num_test):
        bit_train += str(int(x[i]+bias))+str(int(y[i]+bias))
    # 保存数据
    file_name = 'KEY_LogisticHyperChaotic'
    with open(f"{saving_dir_dict[working_loc]}/{file_name}", 'w') as file:
        file.write(bit_train)