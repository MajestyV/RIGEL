# This code is designed for simulating activation function provide by hardware-implemented reservoir neuron.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reservoir neuron based on source follower
def ReservoirNeuron_SourceFollower(Vin):
    Vout = np.piecewise(Vin, [Vin>3.37,Vin<=3.37,Vin<=1.53],
                        [lambda x:1.8, lambda x:0.97026*x-1.49673, lambda x:0])
    return Vout

# For numpy array
def ReservoirNeuron_SourceFollower_matrix(Vin_array):
    Vout_array = np.empty(Vin_array.shape)

    Vout = np.piecewise(Vin_array.getA()[0], [Vin_array.getA()[0]>3.37,Vin_array.getA()[0]<=3.37,Vin_array.getA()[0]<=1.53],
                        [lambda x:1.8, lambda x:0.97026*x-1.49673, lambda x:0])
    Vout_array[0] = Vout
    return np.mat(Vout_array)

# Reservoir neuron based on source follower
def ReservoirNeuron_SourceFollower_central(Vin):
    Vout = np.piecewise(Vin, [Vin > 0.897, Vin <= 0.897, Vin <= -0.961],
                        [lambda x: 1.8, lambda x: 0.97026 * x + 0.92967678, lambda x: 0])
    return Vout

# For numpy array
def ReservoirNeuron_SourceFollower_central_matrix(Vin_array):
    Vout_array = np.empty(Vin_array.shape)

    Vout = np.piecewise(Vin_array.getA()[0], [Vin_array.getA()[0]>0.897,Vin_array.getA()[0]<=0.897,Vin_array.getA()[0]<=-0.961],
                        [lambda x:1.8, lambda x:0.97026 * x + 0.92967678, lambda x:0])
    Vout_array[0] = Vout
    return np.mat(Vout_array)

########################################################################################################################

# Reservoir neuron based on source follower, calculated by GPRA
def SourceFollower_DevVer(Vin: int or float or np.ndarray or np.matrix, Vdd_cond: str='Vdd_1.6V',
                          circuit_param: list=(1.0,1.0), critical_point: float=0.1, centralized: bool=True,
                          shifting: float=3.0) -> float or np.ndarray or np.matrix:
    '''
    共漏电路（源极跟随器）唯象物理模型（用于拟合非线性电路）
    :param Vin: 输入电压
    :param circuit_param: 电路参数，这里是一个列表，第一个元素是 R_i，第二个元素是 R_f
    :param critical_point:  测试临界点（就是测试时取的下界），是一个修正系数
    :return:
    '''

    # 电路参数，这里是一个字典，key是Vdd的电压，value是对应的参数，由GPRA算法得到
    params_dict = {'Vdd_0.0V': [-0.034431, -0.129005, 0.325564, 0.14935, 0.075479, 0.260627],
                   'Vdd_0.2V': [0.130454, 0.945272, 1.640328,  0.619087, -0.136091, 0.196497],
                   'Vdd_0.4V': [0.399816, 0.329577, 2.165105, -0.343039, -0.191711, 0.391775],
                   'Vdd_0.6V': [0.287189, 0.177033, 2.566068, -0.303062, -0.119475, 0.587424],
                   'Vdd_0.8V': [0.297435, 0.120709, 2.761348, -0.571165, -0.085234, 0.780017],
                   'Vdd_1.0V': [0.10261, 0.110385, 2.758942, 0.242143, -0.057671, 0.960984],
                   'Vdd_1.2V': [0.147948, 0.130599, 2.559626, -0.308005, -0.044369, 1.106703],
                   'Vdd_1.4V': [0.156518, 0.140924, 2.488891, -0.409925, -0.041237, 1.147826],
                   'Vdd_1.6V': [0.100847, 0.15066,  2.444929,  -0.037348, -0.035341, 1.159662],
                   'Vdd_1.8V': [0.111536, 0.133681, 2.520653, -0.046825, -0.042997, 1.16394],
                   'Vdd_2.0V': [0.149066, 0.146132, 2.470175, -0.394187, -0.035634, 1.169046]}

    params = params_dict[Vdd_cond]  # 选择对应Vdd电压的参数

    # 由于同相加法器的特点，需先对输出进行scaling
    scaling_factor = 1.0 + circuit_param[0] / circuit_param[1]
    Vin = Vin * scaling_factor

    if centralized:  # 中心化激活函数
        Vin = Vin + shifting

    if isinstance(Vin, (float, int)):  # 如果输入是单个数值，或者是数组

        # 利用np.piecewise()函数实现分段函数，区分大于critical_point和小于等于critical_point的两段函数
        # 器件的模型采用 Stretched expotential function （Exp指数转移特性）
        # 详情请参考: https://en.wikipedia.org/wiki/Stretched_exponential_function
        # critical_point = 0.1  # 测试临界点
        conduct_coeff = np.piecewise(Vin, [Vin > critical_point, Vin <= critical_point],
                                    [lambda x: params[0] * np.exp(params[1] * x ** params[2] + params[3]) +params[4],
                                    lambda x: params[0] * np.exp(params[1] * critical_point ** params[2]+ params[3])+params[4]])


        Vout = params[5] * (1.0 - 1.0 / (1.0 + conduct_coeff))  # 共漏电路（源极跟随器）的电压转移特性

        return Vout

    elif isinstance(Vin, (np.ndarray, np.matrix)):  # 如果输入是矩阵
        Vout_array = np.empty(Vin.shape)  # 创建一个与Vin相同大小的数组

        conduct_coeff = np.piecewise(Vin.getA()[0], [Vin.getA()[0] > 0, Vin.getA()[0] <= 0],
                                     [lambda x: params[0] * np.exp(params[1] * x ** params[2] + params[3]) + params[4],
                                      lambda x: params[0] * np.exp(params[1] * critical_point ** params[2]+ params[3])+params[4]])

        Vout = params[5] * (1.0 - 1.0 / (1.0 + conduct_coeff))

        Vout_array[0] = Vout  # 将Vout赋值给Vout_array的第一个元素

        return np.mat(Vout_array)  # 返回一个矩阵

    else:
        raise ValueError('The input type is not supported!')

# Reservoir neuron based on source follower, calculated by GPRA
def SourceFollower_HIT(Vin: int or float or np.ndarray or np.matrix, Vdd_cond: str='Vdd_1.6V',
                       circuit_param: list=(1.0,1.0), infimum: float=0.1, supremum: float=5.0,
                       centralized: bool=True, shifting: float=3.0) -> float or np.ndarray or np.matrix:
    '''
    共漏电路（源极跟随器）唯象物理模型（用于拟合非线性电路）
    :param Vin: 输入电压
    :param circuit_param: 电路参数，这里是一个列表，第一个元素是 R_i，第二个元素是 R_f
    :param infimum:  测试临界点之一，就是测试时取的下界
    :param supremum: 测试临界点之二，就是测试时取的上界
    :return:
    '''

    # 电路参数，这里是一个字典，key是Vdd的电压，value是对应的参数，由GPRA算法得到
    params_dict = {'Vdd_0.0V': [-0.034431, -0.129005, 0.325564, 0.14935, 0.075479, 0.260627],
                   'Vdd_0.2V': [0.130454, 0.945272, 1.640328,  0.619087, -0.136091, 0.196497],
                   'Vdd_0.4V': [0.399816, 0.329577, 2.165105, -0.343039, -0.191711, 0.391775],
                   'Vdd_0.6V': [0.287189, 0.177033, 2.566068, -0.303062, -0.119475, 0.587424],
                   'Vdd_0.8V': [0.297435, 0.120709, 2.761348, -0.571165, -0.085234, 0.780017],
                   'Vdd_1.0V': [0.10261, 0.110385, 2.758942, 0.242143, -0.057671, 0.960984],
                   'Vdd_1.2V': [0.147948, 0.130599, 2.559626, -0.308005, -0.044369, 1.106703],
                   'Vdd_1.4V': [0.156518, 0.140924, 2.488891, -0.409925, -0.041237, 1.147826],
                   'Vdd_1.6V': [0.100847, 0.15066,  2.444929,  -0.037348, -0.035341, 1.159662],
                   'Vdd_1.8V': [0.111536, 0.133681, 2.520653, -0.046825, -0.042997, 1.16394],
                   'Vdd_2.0V': [0.149066, 0.146132, 2.470175, -0.394187, -0.035634, 1.169046]}

    params = params_dict[Vdd_cond]  # 选择对应Vdd电压的参数

    # 由于同相加法器的特点，需先对输出进行scaling
    scaling_factor = 1.0 + circuit_param[0] / circuit_param[1]
    Vin = Vin * scaling_factor

    if centralized:  # 中心化激活函数
        Vin = Vin + shifting

    # 利用np.piecewise()函数实现分段函数，区分大于critical_point和小于等于critical_point的两段函数
    # 器件的模型采用 Stretched expotential function （Exp指数转移特性）
    # 详情请参考: https://en.wikipedia.org/wiki/Stretched_exponential_function
    conduct_coeff = np.piecewise(Vin, [Vin < infimum, (Vin >= infimum) & (Vin < supremum), Vin >= supremum],
                                 [lambda x: params[0] * np.exp(params[1] * infimum ** params[2] + params[3]) + params[4],
                                  lambda x: params[0] * np.exp(params[1] * x ** params[2] + params[3]) + params[4],
                                  lambda x: params[0] * np.exp(params[1] * supremum ** params[2] + params[3]) + params[4]
                                  ])

    Vout = params[5] * (1.0 - 1.0 / (1.0 + conduct_coeff))  # 共漏电路（源极跟随器）的电压转移特性

    return Vout


if __name__=='__main__':
    # 数据/图像保存文件夹
    # saving_directory = 'C:/Users/DELL/Desktop/ResNode/Working_dir/Demo'  # Lingjiang
    # saving_directory = 'D:/OneDrive/OneDrive - The Chinese University of Hong Kong/Desktop/ResNode/Working_dir/Demo'  # MMW405

    Vdd_list = ['Vdd_0.0V', 'Vdd_0.2V', 'Vdd_0.4V', 'Vdd_0.6V', 'Vdd_0.8V', 'Vdd_1.0V', 'Vdd_1.2V', 'Vdd_1.4V', 'Vdd_1.6V', 'Vdd_1.8V', 'Vdd_2.0V']
    num_Vdd = len(Vdd_list)  # 电源电压的个数

    blues = sns.cubehelix_palette(num_Vdd, rot=-.25, light=.7)  # 蓝调
    # blues = sns.color_palette("Blues", num_Vdd)  # 蓝调
    # blues = sns.color_palette("PuBuGn_d", num_Vdd)  # 蓝调

    # sns.set_palette(blues)  # 设置调色板
    # sns.set_palette(sns.color_palette("PuBuGn_d"))

    x = np.linspace(-5,5, 100)  # 输入电压
    for i in range(num_Vdd):
        y_GPRA = SourceFollower_HIT(x,Vdd_list[i])      # GPRA
        # y_pspice = SourceFollower_CadencePspice(x,Vdd_list[i])  # Cadence Pspice

        plt.plot(x, y_GPRA, label=f'GPRA_{Vdd_list[i]}', color=blues[i])
        # plt.plot(x, y_pspice, label=f'Pspice_{Vdd_list[i]}', color=blues[i], linestyle='--')

    # plt.xlim(0, 5.0)

    # plt.savefig(saving_directory + '/Nonlinearity.eps', dpi=300)  # 保存图像

    plt.show(block=True)  # 显示图像