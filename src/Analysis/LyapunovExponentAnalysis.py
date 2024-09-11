# 此函数包可以对一段时间序列（time series data）进行Lyapunov指数分析，详情请参考：
# This code follows https://blog.csdn.net/itnerd/article/details/107015164.
# https://github.com/manu-mannattil/nolitsa/tree/master

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nolitsa import data, lyapunov

# 此函数可以通过Rosenstein’s algorithm计算时间序列对数差分距离均值谱拟合得到最大Lyapunov指数（MLE - Maximum Lyapunov Exponent）
# 关于原理，可以参考文献：https://www.sciencedirect.com/science/article/pii/016727899390009P
# 以及Julia包：https://juliadynamics.github.io/DynamicalSystems.jl/v1.5/chaos/lyapunovs/
# mean_period参数与Theiler window相关，需要一个合适的Theiler window才能较为准确地计算MLE
def MaximumLyapunovExponent(time_series,mean_period=10,num_step=1000,estimate_steps=200,step_length=0.01):
    time_series_T = time_series.T  # 先转置数据，方便后续计算

    log_div_average = lyapunov.mle(time_series_T, maxt=num_step, window=mean_period)  # 利用Nolitsa计算时间序列的对数差分距离均值谱

    time = np.arange(num_step)*step_length
    coefs = np.polyfit(time[:estimate_steps],log_div_average[:estimate_steps],1)  # 通过线性拟合计算对数差分距离均值谱的初始斜率

    MLE = coefs[0]

    # 可以通过随机抽样一致算法（RANSAC - Random sample consensus）拟合正确的初始曲线
    # https://zh.wikipedia.org/wiki/%E9%9A%A8%E6%A9%9F%E6%8A%BD%E6%A8%A3%E4%B8%80%E8%87%B4
    fitting_result = coefs[1]+coefs[0]*time  # 拟合结果

    return MLE, time, log_div_average, fitting_result

# 此函数专用于对目标系统以及算法模型的拟合结果进行快速的李雅普诺夫指数分析
def Quick_LyapunovAnalysis(ground_truth, network_output, num_step, step_length, mean_period,
                           estimate_steps_truth, estimate_steps_network,
                           filename,saving_directory):
    # 计算目标系统的最大李雅普诺夫指数
    MLE, t, log_div_avg, fitting = MaximumLyapunovExponent(ground_truth, mean_period=mean_period, num_step=num_step,
                                                           estimate_steps=estimate_steps_truth, step_length=step_length)
    # 计算算法拟合结果的最大李雅普诺夫指数
    MLE_network, t_network, log_div_avg_network, fitting_network = MaximumLyapunovExponent(network_output,
                                                                                           mean_period=mean_period,
                                                                                           num_step=num_step,
                                                                                           estimate_steps=estimate_steps_network,
                                                                                           step_length=step_length)

    # 利用pandas保存李雅普诺夫分析结果
    data_DataFrame = pd.DataFrame({'time': t, 'log_div_avg': log_div_avg, 'linear_fit': fitting,
                                   'log_div_avg_ESM': log_div_avg_network, 'linear_fit_ESM': fitting_network})
    data_DataFrame.to_csv(saving_directory + '/'+filename+'.csv', index=False, sep=',')  # 保存为csv文件

    # 画图模块
    print('MLE = ', MLE, ', ', 'MLE_network = ', MLE_network)
    # 原模型
    plt.plot(t, log_div_avg, label='target system', color='b')
    plt.plot(t, fitting, '--', label='Linear fit', color='b')
    # 网络
    plt.plot(t, log_div_avg_network, label='network output', color='r')
    plt.plot(t, fitting_network, '--', label='Linear fit', color='r')

    plt.title('Maximum Lyapunov exponent by Rosenstein’s method')
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Average logarithmic divergence $\langle \mathrm{ln}(d_i(t)) \rangle$')

    plt.legend(frameon=False)
    plt.show(block=True)

    return

if __name__=='__main__':
    dt = 0.01
    x0 = [0.62225717, -0.08232857, 30.60845379]
    x = data.lorenz(length=4000, sample=dt, x0=x0,
                    sigma=16.0, beta=4.0, rho=45.92)[1]

    # print(x)
    print(x.shape)
    # plt.plot(range(len(x)),x)
    # plt.show()

    # Choose appropriate Theiler window.
    mean_period = 10
    num_step = 1000
    estimate_steps = 200

    MLE, t, log_div_avg, fitting = MaximumLyapunovExponent(x.T,mean_period=mean_period,num_step=num_step,
                                                           estimate_steps=estimate_steps,step_length=dt)

    print('MLE = ', MLE)  # MLE - Maximum Lyapunov Exponents

    # 画图模块
    plt.plot(t, log_div_avg, label='divergence', color='k')
    plt.plot(t, t * 1.50, '--', label='slope=1.5', color='r')
    # RANSAC - Random sample consensus, 随机抽样一致 (https://zh.wikipedia.org/wiki/%E9%9A%A8%E6%A9%9F%E6%8A%BD%E6%A8%A3%E4%B8%80%E8%87%B4)
    plt.plot(t, fitting, '--', label='RANSAC', color='b')

    plt.title('Maximum Lyapunov exponent for the Lorenz system')
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Average logarithmic divergence $\langle \mathrm{ln}(d_i(t)) \rangle$')

    plt.legend()
    plt.show(block=True)