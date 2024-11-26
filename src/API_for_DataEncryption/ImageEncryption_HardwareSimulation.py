# This code is designed to designate echo state network to general chaotic time series for image encryption based on the
# general synchronization property of reservoir computing.

import copy
import numpy as np
from src import dynamics, Activation, ESN, Dataset_makeup  # 直接调用，提高运行效率

# from ..Benchmark_DynamicalSystems import DynamicalSystems
# from ..OrganizingDataset import Dataset_makeup

class ImageEncryption_HardwareSimulation:
    ''' 此类函数专为图像加密而设 '''
    def __init__(self,image=None,image_size=None, encryption_epoch = 1, **kwargs):
        self.image = image                        # 要加密的目标图像文件
        self.image_size = image_size              # 要加密的图像尺寸
        self.encryption_epoch = encryption_epoch  # 加密算法的循环次数

        self.teacher_nstep = kwargs['teacher_nstep'] if 'teacher_nstep' in kwargs else 5000   # 教师混沌系统步数
        self.student_nstep = kwargs['student_nstep'] if 'student_nstep' in kwargs else 10000  # ESN训练后自迭代的步数/8

        self.teacher_param = kwargs['teacher_param'] if 'teacher_param' in kwargs else (0.5,3.8)  # 教师混沌系统的参数

        # 用于学习混沌序列的Echo State Network的参数
        # 决定是否丢弃初值，防止初始值影响下后续迭代演化出现偏移
        self.transient = kwargs['transient'] if 'transient' in kwargs else 0  # 默认不丢弃初值（此输入应为整型int）
        # Reservoir neuron setting
        self.activation = kwargs['activation'] if 'activation' in kwargs else Activation.I_Taylor  # 硬件激活函数
        self.bias = kwargs['bias'] if 'bias' in kwargs else 0  # 阈值（threshold）
        self.leaky_rate = kwargs['leaky_rate'] if 'leaky_rate' in kwargs else 1.0  # 确定是否开启Leaky-integrator模式
        # Reservoir topology
        self.res_dim = kwargs['res_dim'] if 'res_dim' in kwargs else 200
        self.res_den = kwargs['res_den'] if 'res_den' in kwargs else 0.1
        self.spec_rad = kwargs['spec_rad'] if 'spec_rad' in kwargs else 1.5
        # Signal scaling factors
        self.input_scaling = kwargs['input_scaling'] if 'input_scaling' in kwargs else 2.0
        self.feedback_scaling = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 0
        self.noise_scaling = kwargs['noise_scaling'] if 'noise_scaling' in kwargs else 0
        self.regularization_coeff = kwargs['regularization_coeff'] if 'regularization_coeff' in kwargs else 1.0
        # 权重优化算法
        self.opt_algorithm = kwargs['opt_algorithm'] if 'opt_algorithm' in kwargs else 4  # 默认为模式4

    # 利用ESN学习混沌序列来生成新的混沌序列
    def GenChaoticBit_logistic(self):
        # 导入logistics函数
        x0, r = self.teacher_param  # 解压教师系统的参数
        time, data = dynamics.DynamicalSystems.Logistic_map(x0=x0, r=r, num_step=self.teacher_nstep)

        # 分割教师混沌系统的数据集
        init_set, training_set, predicting_set = Dataset_makeup(time, data,
                                                                      num_init=int(self.teacher_nstep*0.2),
                                                                      num_train=int(self.teacher_nstep*0.35),
                                                                      num_test=int(self.teacher_nstep*0.35))
        t_train, x_train, y_train = training_set

        # 定义ESN网络
        model = ESN.Analog_ESN(input_dimension=1, output_dimension=1, activation=self.activation,
                               input_scaling=self.input_scaling,
                               reservoir_dimension=self.res_dim,
                               reservoir_spectral_radius=self.spec_rad,
                               reservoir_density=self.res_den,
                               transient=self.transient, bias=self.bias)

        # opt_algorithm=4的SelectKBest算法有奇效，太过夸张，慎用！！！主要是岭回归（opt_algorithm=2）效果太好！！！
        model.Training_phase(x_train, y_train,opt_algorithm=self.opt_algorithm)  # 训练阶段
        y_predict_ESN, u_state_test, r_state_test = model.Predicting_phase(8*self.student_nstep)  # 外推预测

        # 二进制转八位十进制，展开编写，方便计算，提高运行效率
        bit_flow_decimal = np.zeros((self.student_nstep,1))
        for i in range(self.student_nstep):
            b0 = round(y_predict_ESN[0,i * 8])
            b1 = round(y_predict_ESN[0,i * 8+1])
            b2 = round(y_predict_ESN[0,i * 8+2])
            b3 = round(y_predict_ESN[0,i * 8+3])
            b4 = round(y_predict_ESN[0,i * 8+4])
            b5 = round(y_predict_ESN[0,i * 8+5])
            b6 = round(y_predict_ESN[0,i * 8+6])
            b7 = round(y_predict_ESN[0,i * 8+7])
            bit = int(1*b0+2*b1+4*b2+8*b3+16*b4+32*b5+64*b6+128*b7)
            bit_flow_decimal[i,0] = bit
        # 保存一份二进制比特流数据
        bit_flow_binary = [str(int(y_predict_ESN[0,i]+0.31)) for i in range(y_predict_ESN.shape[1])]  # 加0.5进行四舍五入取整

        return bit_flow_decimal, bit_flow_binary

    # 密钥保存函数
    def SaveKey(self,chaotic_bit_flow,saving_directory,nbit_per_row=32,filename='untitled.txt'):
        saving_address = saving_directory+'/'+filename

        bit_flow_str = ''  # 创建一个空字符串
        for i in range(len(chaotic_bit_flow)):  # 将列表形式的数据写成一个字符串
            bit_flow_str += chaotic_bit_flow[i]

        # mode 1
        # file = open(saving_address,'w')  # 设置文件对象
        # num_rows = int(float(len(bit_flow_str))/float(nbit_per_row))  # 行数
        # for n in range(num_rows):
            # file.write(bit_flow_str[n*nbit_per_row:n*nbit_per_row+nbit_per_row]+'\n')
        # file.close()
        # mode 2
        with open(saving_address,'w') as file:
            file.write(bit_flow_str)

        # np.savetxt(saving_address,chaotic_bit_flow)
        return

    # 循环加密模块，方便用小序列对大图像进行加密
    def GenKeyMap_periodic(self):
        ncol, nrow, nchannel = self.image_size
        random_mapping = np.zeros((ncol,nrow,self.encryption_epoch))

        for n in range(self.encryption_epoch):  # 每个循环生成一组特定的logistics序列
            logistic_series_ESN = self.GenChaoticBit_logistic()
            encrypt_seq_length = len(logistic_series_ESN)

            count = 0  # counting index
            for i in range(ncol):
                for j in range(nrow):
                    # 三个通道应该用三个不同的密钥加密，否则周期性很容易会使得图像得以重构
                    random_mapping[i,j,n] = int(logistic_series_ESN[count%encrypt_seq_length,0]*255)
                    count += 1

        return np.uint8(random_mapping)

    # 利用异或运算进行加密
    def Encrypt(self,key):
        ncol, nrow, nchannel = self.image_size

        encrypted_image = copy.deepcopy(self.image)  # 先将目标图像拷贝给加密图像，然后开始迭代加密

        for n in range(self.encryption_epoch):  # 加密循环
            for i in range(ncol):               # 行
                for j in range(nrow):           # 列
                    for k in range(nchannel):   # 色彩通道
                        encrypted_image[i, j, k] = encrypted_image[i, j, k]^key[i, j, n]
        return encrypted_image

    # 利用异或运算进行解密，为加密过程的逆运算
    def Decrypt(self, encrypted_img, key):
        ncol, nrow, nchannel = self.image_size

        decrypted_image = copy.deepcopy(encrypted_img)  # 先将要解密的图像拷贝出来，然后开始迭代解密

        for n in range(self.encryption_epoch):  # 加密循环
            for i in range(ncol):  # 行
                for j in range(nrow):  # 列
                    for k in range(nchannel):  # 色彩通道
                        decrypted_image[i, j, k] = decrypted_image[i, j, k]^key[i, j, self.encryption_epoch-(n+1)]
        return decrypted_image

if __name__=='__main__':
    pass