# This package is designed for evaluating the performance of the reservoir computing (RC) model in regression tasks.
# 这个函数包专用于评估储层算法在回归任务中的表现

import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Iterable  # 用于类型注解

# 此函数可以计算序列的标准差，以衡量数据的离散程度，输入应为一个二维数组
def Standard_deviation(data):
    N = len(data)  # data length，数据长度
    y_bar = np.sum(data, axis=0)/float(N)  # 计算数据均值
    data_average = np.array([y_bar for i in range(N)])
    sigma = np.sqrt(mean_squared_error(data, data_average))  # 数据与其均值间的方均根误差即为标准差，可以衡量数据的离散程度
    return sigma

# 标准差
def STD(data): return np.std(data)

def Deviation_absolute(ground_truth: Iterable, network_output: Iterable) -> float:
    ''' 
    此函数可计算网络输出与真实值之间的绝对误差 
    绝对误差定义为: deviation(n) = ||y_network(n)-y_truth(n)||_2 (两者向量差的二范数)
    '''
    network_output = np.array(network_output)  # 转换数据类型，确保输入变量为数组
    ground_truth = np.array(ground_truth)
    deviation = np.linalg.norm(network_output - ground_truth, ord=2)  # 二范数
    return deviation

def Deviation_relative(ground_truth: Iterable, network_output: Iterable) -> float:
    ''' 
    此函数可计算网络输出与真实值之间的相对误差
    相对误差定义为: deviation(n) = ||y_network(n)-y_truth(n)||_2 / ||y_truth(n)||_2 (两者向量差的二范数除以真实值的二范数)
    '''
    network_output = np.array(network_output)  # 转换数据类型，确保输入变量为数组
    ground_truth = np.array(ground_truth)
    deviation = np.linalg.norm(network_output - ground_truth, ord=2) / np.linalg.norm(ground_truth, ord=2)  # 二范数
    return deviation

# 这个函数可以利用scikit-learn包计算均方误差（Mean Squared Error）
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def MSE(ground_truth,network_output): return mean_squared_error(ground_truth,network_output)

# 计算方均根误差（Root Mean Squared Error）
def RMSE(ground_truth,network_output): return np.sqrt(mean_squared_error(ground_truth,network_output))

# 计算归一化方均根误差（Normalized Root Mean Squared Error）
def NRMSE(ground_truth,network_output):
    RMSE = np.sqrt(mean_squared_error(ground_truth,network_output))  # 计算真实数据与模型预测结果间的方均根误差

    N = len(ground_truth)  # data length，数据长度
    y_bar = np.sum(ground_truth, axis=0) / float(N)  # 计算数据均值
    truth_average = np.array([y_bar for i in range(N)])
    sigma = np.sqrt(mean_squared_error(ground_truth, truth_average))  # 计算标准差，用于归一化

    return RMSE/sigma

# 计算NRMSE，参考：https://ieeexplore.ieee.org/document/9643536
def NRMSE_ICCAD(ground_truth,network_output):
    RMSE = np.sqrt(mean_squared_error(ground_truth,network_output))  # 计算真实数据与模型预测结果间的方均根误差

    N = len(ground_truth)  # data length，数据长度
    # y_bar = np.sum(ground_truth, axis=0) / float(N)  # 计算数据均值
    # truth_average = np.array([y_bar for i in range(N)])
    # sigma = np.std(ground_truth)  # 计算标准差，用于归一化
    y_bar = np.sum(ground_truth, axis=0) / float(N)  # 计算数据均值

    return RMSE/y_bar[0]

# 这个函数可以计算NRMSE (normalized root-mean-square error, 正规化方均根误差)
# 假设输出为长度为P的向量：y(t) = [y1(t), y2(t), ..., yP(t)]
# ground_truth的格式[[y1_target(t0), ...yP_target(t0)], ..., [y1_target(tn), ...yP_target(tn)]]
# predicting的格式[[y1_predict(t0), ...yP_predict(t0)], ..., [y1_predict(tn), ...yP_predict(tn)]]
def NRMSE_homemade(network_output, ground_truth):
    data_length = len(ground_truth)
    network_output = np.array([np.array(network_output[n]) for n in range(data_length)])  # 数据转换
    ground_truth = np.array([np.array(ground_truth[n]) for n in range(data_length)])  # 数据转换
    y_target_average = np.sum(ground_truth, axis=0)/float(data_length)  # 将ground_truth的每一行加起来取平均
    print(y_target_average)

    #y = np.sum(ground_truth,axis=0)
    #print(y)

    network_output_deviation = []  # 预测值对于真实值的偏差
    ground_truth_deviation = []  # 真实值本身的离散度
    for i in range(data_length):
        p_value = np.linalg.norm(network_output[i] - ground_truth[i], ord=2)  # 二范数
        network_output_deviation.append(p_value ** 2)  # 二范数的平方

        g_value = np.linalg.norm(ground_truth[i] - y_target_average, ord=2)
        ground_truth_deviation.append(g_value ** 2)

    network_output_deviation = np.array(network_output_deviation)  # 转换成数组
    ground_truth_deviation = np.array(ground_truth_deviation)

    nrmse = np.sqrt(np.sum(network_output_deviation)/np.sum(ground_truth_deviation))  # 求和相除再开根号

    return nrmse