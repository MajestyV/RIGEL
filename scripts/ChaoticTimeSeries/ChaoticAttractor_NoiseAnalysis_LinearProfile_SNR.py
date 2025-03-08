import os
import numpy as np
import matplotlib.pyplot as plt
from src import Dynamics, Activation, ESN, VISION, Dataset_makeup, Evaluation

from tqdm import tqdm  # 进度条展示

working_loc = 'Lingjiang'

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/NonlinearNode/Simulation/RIGEL/Working_dir'}

def Gen_Lorenz63_w_Noise_Dataset(origin: tuple=(3, 2, 16), parameter: tuple=(10, 28, 8.0 / 3.0), num_step: int=10001,
                                 step_length: float=0.01, num_init: int=2000, num_train: int=4000, num_test: int=3000,
                                 noise_type: str='normal', noise_dist_param: float or tuple=None, SNR: float=200.)\
                                 -> tuple[tuple,tuple,tuple]:
    '''
    生成带噪声的Lorenz63数据集
    '''

    # 导入Lorenz63的数据集
    time, data = Dynamics.Lorenz_63(origin=origin, parameter=parameter, num_step=num_step, step_length=step_length)

    data_w_noise = Dynamics.Add_noise(data, noise_type=noise_type, SNR=SNR, noise_dist_param=noise_dist_param)  # 加入噪声

    dataset = Dataset_makeup(time, data_w_noise,num_init=num_init, num_train=num_train, num_test=num_test)  # 制作数据集

    return dataset

def Reconstructing_by_AnalogESN(dataset: tuple[tuple,tuple,tuple], ESN_type: str='Analog-ESN', input_scaling: float=0.01,
                                activation: callable=np.tanh, leaky_rate: float=None, reservoir_dim: int=400,
                                reservoir_den: float=0.1, spectral_radius: float=1.5, transient: int=1000, bias: float=0,
                                opt_algorithm: int=2) -> tuple:
    '''
    利用ESN进行动态系统重构
    '''

    # 解压数据集
    t_init, x_init, y_init = dataset[0]
    t_train, x_train, y_train = dataset[1]
    t_test, x_test, y_test = dataset[2]

    # 获取数据集的参数
    num_init, num_train, num_test = t_init.shape[0], t_train.shape[0], t_test.shape[0]  # 各阶段的数据子集长度
    input_dim, output_dim = x_train.shape[0], y_train.shape[0]                          # 输入和输出的维度

    model_pipeline = {'Analog-ESN': ESN.Analog_ESN, 'Analog-LiESN': ESN.Analog_LiESN}

    model = model_pipeline[ESN_type](input_dimension=input_dim, output_dimension=output_dim,
                                     input_scaling=input_scaling,
                                     activation=activation,
                                     leaky_rate=leaky_rate,
                                     reservoir_dimension=reservoir_dim,
                                     reservoir_density=reservoir_den,
                                     reservoir_spectral_radius=spectral_radius,
                                     transient=transient,
                                     bias=bias)

    # opt_algorithm=4的SelectKBest算法有奇效，太过夸张，慎用！！！主要是岭回归（opt_algorithm=2）效果太好！！！
    y_train_ESN, y_train, u_state_train, r_state_train, W_out = model.Training_phase(x_train, y_train,
                                                                                     opt_algorithm=opt_algorithm)
    # 此模型可以利用transient参数先把前面一段储层的初始态去掉
    t_train_new = np.array([i + num_init + transient for i in range(y_train.shape[1])])

    y_test_ESN, u_state_test, r_state_test = model.Predicting_phase(num_test)

    return (t_train_new, y_train, y_train_ESN), (t_test, y_test, y_test_ESN)

if __name__=='__main__':
    print('Start noise analysis process...')

    noise_type, noise_dist_param, npoints_eval = ('Poisson', 10, 1000)  # 设置噪声测试的噪声类型和信噪比

    for ESN_param in [('Analog-ESN', 'FSJ'), ('Analog-LiESN', 'FSJ'), ('Analog-ESN', 'FSJ-clipped'), ('Analog-LiESN', 'FSJ-clipped')]:
        ESN_type, activation_type = ESN_param  # 设置ESN模型的参数

        print(f'Start noise analysis linear profile scanning of {ESN_type} with {activation_type} activation...')

        saving_path = f'{saving_dir_dict[working_loc]}/{ESN_type}_{activation_type}_HausdorffDistance_{noise_type}'

        folder = os.path.exists(saving_path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(saving_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        else:
            raise ValueError("Such folder already exists!")

        # Parameter pipeline
        if ESN_type == 'Analog-ESN' and activation_type == 'FSJ':
            def activation(x): return Activation.I_Taylor(x)  # 定义激活函数
            input_scaling = 0.001
            reservoir_dim = 400
            reservoir_den = 0.1
            spectral_radius = 0.9

        elif ESN_type == 'Analog-LiESN' and activation_type == 'FSJ':
            def activation(x): return Activation.I_Taylor(x)  # 定义激活函数
            input_scaling = 0.25
            leaky_rate = 0.95
            reservoir_dim = 400
            reservoir_den = 0.05
            spectral_radius = 1.2

        elif ESN_type == 'Analog-ESN' and activation_type == 'FSJ-clipped':
            def activation(x): return Activation.I_Taylor_w_OperationalRange(x, operational_range=(-3, 3))  # 定义激活函数
            input_scaling = 0.01
            reservoir_dim = 400
            reservoir_den = 0.1
            spectral_radius = 1.5

        elif ESN_type == 'Analog-LiESN' and activation_type == 'FSJ-clipped':
            def activation(x): return Activation.I_Taylor_w_OperationalRange(x, operational_range=(-3, 3))  # 定义激活函数
            input_scaling = 0.25
            leaky_rate = 0.95
            reservoir_dim = 400
            reservoir_den = 0.05
            spectral_radius = 1.2

        else:
            raise ValueError('Invalid ESN type or activation type!')

        # 进行线扫描
        d_Hausdorff_list = []  # 创建一个空列表存放结果
        for SNR in tqdm(range(10, 210, 10)):

            # 生成带噪声的Lorenz63数据集
            dataset = Gen_Lorenz63_w_Noise_Dataset(noise_type=noise_type, SNR=SNR, noise_dist_param=noise_dist_param)

            # 利用ESN进行动态系统重构
            (t_train_new, y_train, y_train_ESN), (t_test, y_test, y_test_ESN) = Reconstructing_by_AnalogESN(dataset, ESN_type=ESN_type, activation=activation)

            d_Hausdorff = Evaluation.Hausdorff_distance(y_test, y_test_ESN, npoints_eval=npoints_eval)

            d_Hausdorff_list.append([SNR,d_Hausdorff])

            # 网络拟合结果分析
            VISION.Analyze_3D_systems(np.hstack((t_train_new, t_test)), (y_train, y_train_ESN), (y_test, y_test_ESN),
                                      deviation_range=(0, 100))

            # 保存数据
            for fmt in ['eps', 'png', 'pdf']:
                plt.savefig(f'{saving_path}/{ESN_type}_{activation_type}_NoiseAnalysis_{noise_type}_SNR-{SNR}.{fmt}', format=fmt)

        # 保存 Hausdorff 距离数据
        np.savetxt(f'{saving_path}/{ESN_type}_{activation_type}_HausdorffDistance_{noise_type}.txt', np.array(d_Hausdorff_list))

    print('Noise analysis process finished!')
