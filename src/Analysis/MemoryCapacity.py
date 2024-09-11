# This code is designed to characterize the short-term memory capacity of recurrent neural network (RNN).
import os
import numpy as np
import pandas as pd
import ASHEN as an
import matplotlib.pyplot as plt

# 此函数专为从Dataset文件夹中读出memory capacity任务所需的数据文件而设计
def Get_MemoryCapacity_Dataset():
    default_data_directory = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "../Dataset")  # 默认数据文件夹
    default_filename = 'memory_capacity_100.csv'  # 默认数据文件名

    data_file = default_data_directory + '/' + default_filename

    print(data_file)

    data_DataFrame = pd.read_csv(data_file, header=None, skiprows=19, delimiter=',')  # 利用skiprows跳过前19行注释行

    data_array = data_DataFrame.values

    print(data_array.shape)

    x, y = (data_array[:, 0], data_array[:, 1:])
    print(y.shape)
    return x, y

# 此函数专为计算非线性RNN的short-term memory而设计
# This function is designed for examining the short term memory capacity of reservoir computing models.
# Following the work of H. Jaeger: https://www.ai.rug.nl/minds/uploads/STMEchoStatesTechRep.pdf
# The input signal is stationary, as such the variance is time-independent: var[x(t)] = var[x(t-tau)]
# The input signal and the network reconstruction should have the shape (n_channel, data_length),
# where n_channel is the number of signal channels and data_length is the data series length.
# In the most original definition, the input and network reconstruction should be single channel that n_channel=1.
def MemoryCapacity(input_signal, network_reconstruction, tau):
    n_channel, data_length = input_signal.shape
    MC = 0
    for i in range(n_channel):
        corrcoeff = np.corrcoef(input_signal[i,-tau:], network_reconstruction[i,-tau:])
        MC += (corrcoeff[0,1])**2  # Memory capacity的定义即为相关系数的平方
    MC = MC/float(n_channel)  # 取多通道均值
    return MC

if __name__=='__main__':
    # 准备Memory capacity测试所用的数据集
    random_seed = 2048  # 用于生成（伪）随机数的种子

    num_channel = 1
    data_length = 20000

    tau = 100  # 时延参数

    rng = np.random.RandomState(random_seed)  # 根据种子产生随机数列

    data_initial = rng.rand(num_channel, data_length)
    # 将数据集分割成训练集和测试集
    train_data = data_initial[:, tau+1:(16000+tau+1)]
    test_data = data_initial[:, (16000+tau+1):]

    memory_capacity = np.empty((tau))  # 创建空数组来存放不同时延的memory capacity

    # 调用ESN网络
    ESN = an.ESN_HIT(input_dimension=1, output_dimension=1,
                     input_scaling=0.1,
                     activation=np.tanh,
                     reservoir_dimension=60,
                     reservoir_spectral_radius=1.0,
                     # reservoir_connection_weight=W_res,
                     transient=1000,
                     bias=0)

    for i in range(tau, 0, -1):
        # print(i)
        Expect_output = data_initial[:, i:(16000 + (i))]
        # print(Expect_output.shape)
        y_train_ESN, y_train, r_state_train, W_out = ESN.Training_phase(train_data, Expect_output, opt_algorithm=4)

        # print(pred_train.shape, Expect_output.shape)
        memory_capacity[tau-i] = MemoryCapacity(y_train_ESN, y_train, tau=tau)
        print(memory_capacity[tau - i])

    print(np.sum(memory_capacity[memory_capacity > 0.01]))

    plt.plot(memory_capacity)
    plt.show(block=True)