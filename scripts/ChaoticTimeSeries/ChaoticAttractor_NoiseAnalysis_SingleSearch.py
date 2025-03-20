import numpy as np
import matplotlib.pyplot as plt
from src import Dynamics, Activation, ESN, VISION, Dataset_makeup, Evaluation

from tqdm import tqdm  # 进度条展示

working_loc = 'iSense'

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/NonlinearNode/Simulation/RIGEL/Working_dir',
                   'iSense': 'F:/DZTT滴磕盐资料/磕盐项目/NonlinearNode/Simulation/RIGEL/Noise_perturbation/SingleSearch'}

def Gen_Lorenz63_w_Noise_Dataset(origin: tuple=(3, 2, 16), parameter: tuple=(10, 28, 8.0 / 3.0), num_step: int=10001,
                                 step_length: float=0.01, num_init: int=2000, num_train: int=4000, num_test: int=3000,
                                 noise_type: str='normal', noise_dist_param: float or tuple=None, SNR: float=200.) \
                                 -> tuple[tuple,tuple,tuple]:
    '''
    生成带噪声的Lorenz63数据集
    '''

    # 导入Lorenz63的数据集
    time, data = Dynamics.Lorenz_63(origin=origin, parameter=parameter, num_step=num_step, step_length=step_length)

    data_w_noise = Dynamics.Add_noise(data, noise_type=noise_type, noise_dist_param=noise_dist_param, SNR=SNR)  # 加入噪声

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

    # 设置噪声的类型、参数、信噪比、评估点数
    # noise_type, noise_dist_param, SNR, npoints_eval = ('uniform', (-0.5,0.5), 20, 1000)  # 均匀分布噪声
    noise_type, noise_dist_param, SNR, npoints_eval = ('Rayleigh', 0.1, 20, 1000)  # 均匀分布噪声
    # noise_type, noise_dist_param, SNR, npoints_eval = ('Poisson', 10, 20, 1000)  # 均匀分布噪声

    # 设置ESN模型的参数
    ESN_type = 'Analog-ESN'  # 设置ESN的类型
    activation_type = 'FSJ'  # 设置激活函数的类型

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

    # 生成带噪声的Lorenz63数据集
    dataset = Gen_Lorenz63_w_Noise_Dataset(noise_type=noise_type, SNR=SNR, noise_dist_param=noise_dist_param)

    # 利用ESN进行动态系统重构
    (t_train_new, y_train, y_train_ESN), (t_test, y_test, y_test_ESN) = Reconstructing_by_AnalogESN(dataset, ESN_type=ESN_type, activation=activation)

    d_Hausdorff_list = []  # 创建一个空列表存放结果
    for npoints_eval in range(100,3100, 100):
        d_Hausdorff = Evaluation.Hausdorff_distance(y_test, y_test_ESN, npoints_eval=npoints_eval)

        d_Hausdorff_list.append([npoints_eval,d_Hausdorff])

    # 保存 Hausdorff 距离数据
    np.savetxt(f'{saving_dir_dict[working_loc]}/{ESN_type}_{activation_type}_HausdorffDistance_{noise_type}_SNR-{SNR}.txt',
               np.array(d_Hausdorff_list))

    # 可视化分析
    VISION.Analyze_3D_systems(np.hstack((t_train_new, t_test)), (y_train, y_train_ESN), (y_test, y_test_ESN),
                              deviation_range=(0, 100))

    # 保存数据
    for fmt in ['eps', 'png', 'pdf']:
        plt.savefig(f'{saving_dir_dict[working_loc]}/{ESN_type}_{activation_type}_NoiseAnalysis_{noise_type}_SNR-{SNR}.{fmt}', format=fmt)

    plt.show(block=True)  # 显示图像
