import codecs  # https://docs.python.org/3/library/codecs.html
import scipy.sparse
import numpy as np
import pandas as pd

# default_directory = path.dirname(__file__) + '/'  # 设置这个代码文件所在的文件夹为默认读写目录

class record:
    """ This code is designed to record the key data of an ESN model in order to analyze its functionality. """
    def __init__(self):
        self.name = record

    # 此函数可以记录神经网络的参数设置
    def SavingNetworkParameters(self,saving_directory, **kwargs):
        data_file = saving_directory+'/ESN_parameters.txt'

        target_system = kwargs['target_system'] if 'target_system' in kwargs else 'N/A'  # 要学习的目标动态系统
        # 回声状态网络的参数
        leaking_rate = kwargs['leaking_rate'] if 'leaking_rate' in kwargs else 'N/A'  # ESN的leaking rate
        input_scaling = kwargs['input_scaling'] if 'input_scaling' in kwargs else 'N/A'  # 输入的缩放因子
        feedback_scaling = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 'N/A'  # 输出反馈的缩放因子

        # 水库矩阵（Reservoir Matrix）的参数
        reservoir_dim = kwargs['reservoir_dim'] if 'reservoir_dim' in kwargs else 'N/A'  # 水库矩阵维数（水库矩阵为一个方阵）
        spectral_radius = kwargs['reservoir_spectral_radius'] if 'reservoir_spectral_radius' in kwargs else 'N/A'  # 水库矩阵的谱半径
        reservoir_density = kwargs['reservoir_density'] if 'reservoir_density' in kwargs else 'N/A'  # 水库矩阵的密度

        # 将网络参数写入数据文件
        file = codecs.open(data_file, 'w')  # 利用codecs中的open函数创建文件，'w'赋予写入权限
        file.write('Target system: '+str(target_system)+'\n'
                   'Leaking rate: '+str(leaking_rate)+'\n'
                   'Input scaling: '+str(input_scaling)+'\n'
                   'Feedback scaling: '+str(feedback_scaling)+'\n'
                   'Reservoir dimension: '+str(reservoir_dim)+'\n'
                   'Reservoir spectral radius: '+str(spectral_radius)+'\n'
                   'Reservoir density: '+str(reservoir_density)+'\n')

        # 激活函数部分
        if 'device_characteristics' or 'device_parameterized' in kwargs:  # 记录是否采用器件参数化激活函数验证ESN硬件化
            file.write('Activation function: Device-parameterized\n')  # 在数据文件中写入激活函数特性
            device_model = kwargs['device_model'] if 'device_model' in kwargs else 'N/A'  # 器件性能的拟合模型
            file.write('Device model: '+str(device_model)+'\n')  # 在数据文件中写入器件模型
            device_parameters = kwargs['device_parameters'] if 'device_parameters' in kwargs else 'N/A'  # 器件性能参数
            file.write('Device parameters: '+str(device_parameters)+'\n')  # 在数据文件中写入器件参数
            reference_factor = kwargs['reference_factor'] if 'reference_factor' in kwargs else '1'  # 信号转换因子，默认为1
            file.write('Reference factor: '+str(reference_factor)+'\n')  # 在数据文件中写入器件参数
        else:
            activation_function = kwargs['activation_function'] if 'activation_function' in kwargs else 'N/A'  # 激活函数
            file.write('Activation function: '+str(activation_function)+'\n')

        file.close()  # 数据写入结束，关闭文件

        return

    # 此函数可以记录输入的权重矩阵
    def SavingInputWeight(self,saving_directory,input_weight):
        data_file = saving_directory+'/Input_Weight_Matrix.txt'

        np.savetxt(data_file, input_weight, delimiter=',', header='Input weight matrix')  # 保存输入权重矩阵到data_file

        return

    # 此函数可以记录水库权重矩阵（水库权重矩阵是一个稀疏矩阵），默认输入未经压缩
    def SavingReservoir(self,saving_directory,reservoir,compressed='False'):
        data_file = saving_directory+'/Reservoir_Weight_Matrix.npz'

        if compressed == 'False':
            reservoir = scipy.sparse.coo_matrix(reservoir)  # 将完全展开的矩阵按照COO格式压缩
            scipy.sparse.save_npz(data_file,reservoir)
        else:
            scipy.sparse.save_npz(data_file, reservoir)

        return

    ################################################################################################################
    # 接下来的函数是专门针对分析Analog-ESN而设计的
    # 此函数可以记录未激活的水库态的范围
    def SavingUnactivatedStatesRange(self,saving_directory,unact_range):
        data_file = saving_directory+'/Unactivated_States_Range.csv'

        # 对数据进行重整，加入时间步一列
        unact_range_rearranged = [[i+1,unact_range[i][0],unact_range[i][1]] for i in range(len(unact_range))]

        data = pd.DataFrame(unact_range_rearranged,columns=['Time_step','Min','Max'])  # 将数据转换为DataFrame格式并加入表头
        data.to_csv(data_file,index=False)  # 不写入行序号

        return

###############################################################################################################
    # 图像保存模块
    # 图像保存函数
    def SavingFigure(self, saving_directory, **kwargs):
        filename = kwargs['filename'] if 'filename' in kwargs else 'Untitled'  # 文件名
        format = kwargs['format'] if 'format' in kwargs else 'eps'  # 储存格式
        dpi = kwargs['dpi'] if 'dpi' in kwargs else 600  # 分辨率

        saving_address = saving_directory+'/'+filename+'.'+format  # 图像文件要储存到的绝对地址

        plt.savefig(saving_address, dpi=dpi, format=format)

        return