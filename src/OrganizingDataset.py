# 直接声明动态系统生成函数，提高运行效率
import numpy as np
from src.DynamicalSystemGenerator import NARMA

# 此函数可以对时间序列进行分割，从而得到Reservoir computing所需的数据集
# 对动态系统数据进行切片以得到我们的训练集跟预测集（应注意，python切片是左闭右开的，如[3:6]只包含下标为3，4，5的）
# 同时，应注意做切片时要用未重整化的数据
# num_intial - 初始化（initialization）阶段，要舍弃的数据（前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值）
# num_train - 训练集长度
# num_predict - 测试集长度
# 最重要的是，由于python的indexing从0开始，所以必须要保证时序信号长度data_length > num_discard+num_train+num_predict
def Dataset_makeup(time: np.ndarray, data_series: np.ndarray, num_init: int, num_train: int, num_test: int):
    # 初始化阶段的数据
    init_start, init_end = [0, num_init]  # 初始化阶段的起点跟终点
    t_init = time[init_start:init_end]
    x_init = data_series[:, init_start:init_end]
    y_init = data_series[:, (init_start + 1):(init_end + 1)]
    initialization_set = (t_init, x_init, y_init)
    # 训练集
    train_start, train_end = [num_init, num_init + num_train]  # 训练集的起点跟终点
    t_train = time[train_start:train_end]
    x_train = data_series[:,train_start:train_end]
    y_train = data_series[:,(train_start+1):(train_end+1)]
    training_set = (t_train,x_train,y_train)
    # 测试集
    test_start, test_end = [num_init + num_train, num_init + num_train + num_test]  # 测试集的起点跟终点
    t_test = time[test_start:test_end]
    x_test = data_series[:,test_start:test_end]
    y_test = data_series[:,(test_start+1):(test_end+1)]
    testing_set = (t_test,x_test,y_test)

    return initialization_set, training_set, testing_set

def Dataset_Makeup4ERA(time: np.ndarray, data_series: np.ndarray, num_init: int, num_train: int, num_valid: int,
                       num_test: int):
    '''
    此函数可以对时间序列进行分割，从而得到ERA模型专用的数据集
    '''

    # 初始化阶段的数据
    init_start, init_end = [0, num_init]  # 初始化阶段的起点跟终点
    t_init = time[init_start:init_end]
    x_init = data_series[:, init_start:init_end]
    y_init = data_series[:, (init_start + 1):(init_end + 1)]
    initialization_set = (t_init, x_init, y_init)
    # 训练集
    train_start, train_end = [num_init, num_init + num_train]  # 训练集的起点跟终点
    t_train = time[train_start:train_end]
    x_train = data_series[:,train_start:train_end]
    y_train = data_series[:,(train_start+1):(train_end+1)]
    training_set = (t_train,x_train,y_train)
    # 验证集
    valid_start, valid_end = [num_init+num_train, num_init + num_train + num_valid]  # 验证集的起点跟终点
    t_vaild = time[valid_start:valid_end]
    x_valid = data_series[:, valid_start:valid_end]
    y_valid = data_series[:, (valid_start + 1):(valid_end + 1)]
    validation_set = (t_vaild, x_valid, y_valid)
    # 测试集
    test_start, test_end = [num_init + num_train + num_valid, num_init + num_train + num_valid + num_test]  # 测试集的起点跟终点
    t_test = time[test_start:test_end]
    x_test = data_series[:,test_start:test_end]
    y_test = data_series[:,(test_start+1):(test_end+1)]
    testing_set = (t_test,x_test,y_test)

    return initialization_set, training_set, validation_set,  testing_set

# 此函数可以用于生成NARMA数据集用于RNN的基线测试
def Gen_NARMA(num_init=2000,num_train=2000,num_test=2000,**kwargs):
    num_step = num_init+num_train+num_test  # 目标系统的运行总步数

    if 'data' in kwargs:
        NARMA_data = kwargs['data']
    else:
        NARMA_data = NARMA.NARMA_10(num_step=num_step)
    time, input, output = NARMA_data

    # 初始化阶段的数据
    init_start, init_end = [0, num_init]  # 初始化阶段的起点跟终点
    t_init = time[init_start:init_end]
    x_init = input[:, init_start:init_end]
    y_init = output[:, init_start:init_end]
    initialization_set = (t_init, x_init, y_init)
    # 训练集
    train_start, train_end = [num_init, num_init + num_train]  # 训练集的起点跟终点
    t_train = time[train_start:train_end]
    x_train = input[:, train_start:train_end]
    y_train = output[:, train_start:train_end]
    training_set = (t_train, x_train, y_train)
    # 测试集
    test_start, test_end = [num_init + num_train, num_init + num_train + num_test]  # 测试集的起点跟终点
    t_test = time[test_start:test_end]
    x_test = input[:, test_start:test_end]
    y_test = output[:, test_start:test_end]
    testing_set = (t_test, x_test, y_test)

    return initialization_set, training_set, testing_set