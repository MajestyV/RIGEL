import numpy as np
import matplotlib.pyplot as plt

# The nonlinear autoregressive moving average (NARMA) task is the baseline test for the performance of recurrent neural
# networks (RNN). For more details, please refer to:
# https://ieeexplore.ieee.org/document/846741
# https://www.arxiv-vanity.com/papers/1906.04608/
# https://arxiv.org/abs/1401.2224

# This function is designed to generate the NARMA-10 time series for the evaluation of RNN models and more
def NARMA_10(initial=0,parameter=(10,0.3,0.05,1.5,0.1),input_range=(0,0.5),num_step=5000):
    n, alpha, beta, gamma, delta = parameter  # 解压系统参数
    u = np.random.uniform(input_range[0],input_range[1],num_step)     # 生成输入序列，默认为一个均匀随机分布

    t = np.zeros(num_step)  # 信号时序
    y = np.zeros((1,num_step))         # NARMA系统的值，为了代码整体规格的统一，这里采用二维张量来存放数据

    for i in range(num_step):
        t[i] = i  # 记录信号时序

        if i < (n - 1):  # 因为python从0开始，所以要减一
            y[0,i] = initial  # 赋予初值
        else:
            longterm = 0  # 长期记忆项
            for j in range(n):
                longterm += y[0,i-j]

            y[0,i] = alpha*y[0,i-1]+beta*y[0,i-1]*longterm+gamma*u[i-(n-1)]*u[i-1]+delta

    return t, y

# This function is designed to generate the NARMA-20 time series
def NARMA_20(initial=0,parameter=(20,0.3,0.05,1.5,0.1),input_range=(0,0.5),num_step=5000):
    n, alpha, beta, gamma, delta = parameter  # 解压系统参数
    u = np.random.uniform(input_range[0],input_range[1],num_step)     # 生成输入序列，默认为一个均匀随机分布

    t = range(num_step)  # 时序信号，从0开始，[0, 1, 2, 3, ...]
    y = np.zeros(num_step)

    for i in range(num_step):
        if i < (n - 1):  # 因为python从0开始，所以要减一
            y[i] = initial  # 赋予初值
        else:
            longterm = 0  # 长期记忆项
            for j in range(n):
                longterm += y[i - j]

            y[i] = np.tanh(alpha*y[i-1]+beta*y[i-1]*longterm+gamma*u[i-(n-1)]*u[i-1]+delta)

    data_set = np.empty((2, num_step))
    data_set[0] = t  # 时序信号，从0开始，[0, 1, 2, 3, ...]
    data_set[1] = y  # NARMA-10 time series

    return data_set

# This function is designed to generate the NARMA-10 time series for the evaluation of RNN models and more

if __name__=='__main__':
    narma_10 = NARMA_10()
    print(narma_10[0])

    # narma_20 = NARMA_20()

    plt.plot(narma_10[0][:,0],narma_10[1][:,0])
    # plt.plot(narma_20[0],narma_20[1])
