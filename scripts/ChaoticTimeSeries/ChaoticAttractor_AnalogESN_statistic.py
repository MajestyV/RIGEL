# 导入环境
import os, sys
current_path = os.path.abspath(os.path.dirname(__file__))  # 获取文件目录
project_path = current_path[:current_path.find('RIGEL') + len('RIGEL')]  # 获取项目根路径，内容为当前项目的名字，即RIGEL
sys.path.append(project_path)  # 将项目根路径添加到系统路径中，以便导入项目中的模块

# result_dir = f'{project_path}/results/Analog-ESN_statistic_result'  # 用于存放结果的目录
result_dir = f'{project_path}/results/Analog-ESN_comparison_with_TimeVaryingActivation'  # 用于存放结果的目录

# 导入所需的库和模块
import numpy as np
import pandas as pd  # 导入pandas库用于数据处理
import seaborn as sns  # 导入seaborn库用于绘图
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
from sklearn.metrics import mean_squared_error, r2_score  # 导入均方误差和R²评分
from tqdm import tqdm  # 进度条库
from src import Dynamics, Activation, ESN, VISION, Dataset_makeup

def define_target():
    num_step = 10001  # 总步数
    step_length = 0.01
    time, data = Dynamics.Lorenz_63(origin=(3,2,16),parameter=(10,28,8.0/3.0),num_step=num_step,step_length=step_length)

    # data = Dynamics.Add_noise(data, SNR=20.)  # 加入噪声

    num_init = 2000  # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
    num_train = 4000  # 训练集长度
    num_test = 3000  # 预测集长度
    result = Dataset_makeup(time, data,num_init=num_init, num_train=num_train, num_test=num_test)

    return result

# def act_func(x): return Activation.I_Taylor_w_OperationalRange(x, operational_range=(-3,3))  # 激活函数，考虑了器件的非理想特性

# def act_func(x): return Activation.I_Taylor(x)  # 激活函数，理想器件

def act_func(x): return Activation.I_Taylor_w_Deviation(x)  # 激活函数，考虑了器件的非理想特性，并且加入了随机偏差

def main():
    ''' 主循环函数 '''
    num_init = 2000  # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
    num_train = 4000  # 训练集长度
    num_test = 3000  # 预测集长度

    initialization_set, training_set, testing_set = define_target()
    
    t_init, x_init, y_init = initialization_set
    t_train, x_train, y_train = training_set
    t_test, x_test, y_test = testing_set

    # NOTE: 定义一些常用的网络参数
    input_scaling = 0.005  # 如果计算结果出现nan，则可以考虑先降低输入的缩放因子，因为我们的激活函数是无界函数，很容易超出计算机所能处理的量程
    leaking_rate = 0.95
    # NOTE: 水库权重矩阵的参数
    reservoir_dim = 400  # N是水库矩库的边长，同时也就是水库态向量的长度
    spectral_radius = 1.0
    reservoir_den = 0.05

    transient = 1000

    model = ESN.Analog_LiESN(input_dimension=3,output_dimension=3,
                             input_scaling=input_scaling,
                             activation=act_func,
                             leaking_rate=leaking_rate,
                             reservoir_dimension=reservoir_dim,
                             reservoir_density=reservoir_den,
                             reservoir_spectral_radius=spectral_radius,
                             transient=transient, bias=0)

    # opt_algorithm=4的SelectKBest算法有奇效，太过夸张，慎用！！！主要是岭回归（opt_algorithm=2）效果太好！！！
    y_train_ESN, y_train, u_state_train, r_state_train, W_out = model.Training_phase(x_train, y_train, opt_algorithm=0)
    # 此模型可以利用transient参数先把前面一段储层的初始态去掉
    t_train_new = np.array([i + num_init + transient for i in range(y_train.shape[1])])

    y_test_ESN, u_state_test, r_state_test = model.Predicting_phase(num_test)

    metric_dict = {}  # 创建一个用于存储各种指标的字典
    for num_samples in [500, 1000, 1500, 2000, 2500, 3000]:
        metric_dict[f'MSE_{num_samples}'] = mean_squared_error(y_test_ESN[:, :num_samples], y_test[:, :num_samples])  # 计算前num_samples个点的均方误差
        metric_dict[f'R2_{num_samples}'] = r2_score(y_test_ESN[:, :num_samples], y_test[:, :num_samples])             # 计算前num_samples个点的R²评分

    return metric_dict

if __name__ == '__main__':
    '''多次运行模型，输出结果，并画统计分布图 '''

    print('Running statistical analysis of the ESN model for ...')
    if not os.path.exists(result_dir):  # 如果结果保存目录不存在，则创建它
        os.makedirs(result_dir)  # 创建目录

    num_cycles = 100  # 运行次数
    
    keys = ['run_index', 'MSE_500', 'R2_500', 'MSE_1000', 'R2_1000', 'MSE_1500', 'R2_1500', 'MSE_2000', 'R2_2000', 'MSE_2500', 'R2_2500', 'MSE_3000', 'R2_3000']
    values = [[] for _ in range(len(keys))]  # 初始化统计结果字典的值为一个空列表，长度与keys相同 (不能直接 [[]]*len(keys) 因为这样会导致所有列表指向同一个对象)
    statistic_result = dict(zip(keys, values))  # 将keys和values组合成一个字典, 用于存储统计结果

    # results = []  # 用于存储每次运行的结果
    for i in tqdm(range(num_cycles), desc='Running ESN model'):
        metrics = main()  # 调用主函数获取指标

        # 将指标添加到统计结果字典中
        statistic_result['run_index'].append(i + 1)  # 记录运行索引
        for key in metrics.keys():
            statistic_result[key].append(metrics[key])

    statistic_result_DF = pd.DataFrame(statistic_result)  # 将统计结果转换为DataFrame
    statistic_result_DF.to_csv(f'{result_dir}/AnalogESN_w_TimeVaryingActivation_{num_cycles}.csv', index=False)  # 保存统计结果到CSV文件

    # sns.displot([result['MSE_1000'] for result in results], kde=True, label='MSE_1000')'

    # for key in statistic_result.keys():
        # print(f'{key}: {np.mean(statistic_result[key]):.4f} ± {np.std(statistic_result[key]):.4f}')

        # result_subdir = f'{result_dir}/{key}'  # 创建一个子目录用于存放每个指标的结果
        # if not os.path.exists(result_subdir):  # 如果子目录不存在，则创建它
            # os.makedirs(result_subdir)
        
        # 将每个指标的结果保存到对应的子目录中
        # for fmt in ['eps', 'png', 'pdf']:


    # sns.displot(statistic_result['R2_1000'], kde=True, label='R2_1000')

    # plt.savefig(f'{result_dir}/Untitled.png')

    print('Statistical analysis completed and results saved to CSV file.')