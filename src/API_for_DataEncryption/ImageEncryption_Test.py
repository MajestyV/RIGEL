# This code is designed to designate echo state network to general chaotic time series for image encryption based on the
# general synchronization property of reservoir computing.

import copy
import ASHEN  # 直接调用，提高运行效率
import numpy as np
# from ..Benchmark_DynamicalSystems import DynamicalSystems
# from ..OrganizingDataset import Dataset_makeup

class ImageEncryption_Test:
    ''' 此类函数专为图像加密而设 '''
    def __init__(self,image,image_size, encryption_epoch = 1, **kwargs):
        self.image = image                        # 要加密的目标图像文件
        self.image_size = image_size              # 要加密的图像尺寸
        self.encryption_epoch = encryption_epoch  # 加密算法的循环次数

        self.teacher_nstep = kwargs['teacher_nstep'] if 'teacher_nstep' in kwargs else 5000   # 教师混沌系统步数
        self.student_nstep = kwargs['student_nstep'] if 'student_nstep' in kwargs else 10000  # ESN训练后自迭代的步数/8

        self.teacher_param = kwargs['teacher_param'] if 'teacher_param' in kwargs else (0.1,3.9)  # 教师混沌系统的参数

        # 用于学习混沌序列的Echo State Network的参数
        self.activation = kwargs['activation'] if 'activation' in kwargs else np.tanh  # 激活函数
        self.res_dim = kwargs['res_dim'] if 'res_dim' in kwargs else 500
        self.leaky_rate = kwargs['leaky_rate'] if 'leaky_rate' in kwargs else 0.9
        self.reservoir_spectral_radius = kwargs['reservoir_spectral_radius'] if 'reservoir_spectral_radius' in kwargs \
                                         else 1.05
        self.input_scaling = kwargs['input_scaling'] if 'input_scaling' in kwargs else 1.0
        self.feedback_scaling = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 0
        self.noise_scaling = kwargs['noise_scaling'] if 'noise_scaling' in kwargs else 0
        self.regularization_coefficient = kwargs['regularization_coefficient'] if 'regularization_coefficient' in \
                                          kwargs else 1.0

    # 利用ESN学习混沌序列来生成新的混沌序列
    def GenChaoticBit_logistic(self):
        # 导入logistics函数
        x0, r = self.teacher_param  # 解压教师系统的参数
        time, data = ASHEN.DynamicalSystems.Logistic_map(x0=x0, r=r, num_step=self.teacher_nstep)

        # 分割教师混沌系统的数据集
        training_set, predicting_set = ASHEN.Dataset_makeup(time, data,
                                                            num_init=int(self.teacher_nstep*0.2),
                                                            num_train=int(self.teacher_nstep*0.35),
                                                            num_test=int(self.teacher_nstep*0.35))
        t_train, x_train, y_train = training_set

        # 定义ESN网络
        ESN = ASHEN.EchoStateNetwork.ESN(x_train, y_train, self.activation,
                                         leaking_rate=self.leaky_rate,
                                         reservoir_dimension=self.res_dim,
                                         reservoir_spectral_radius=self.reservoir_spectral_radius,
                                         input_scaling=self.input_scaling,
                                         feedback_scaling=self.feedback_scaling,
                                         noise_scaling=self.noise_scaling,
                                         regularization_coefficient=self.regularization_coefficient)

        r_train, y_train_ESN, W_out, threshold = ESN.TrainESN()  # 基于训练集计算输出连接权重

        y_predict_ESN = ESN.Forecasting(r_train, W_out, threshold, 8*self.student_nstep)  # 利用训练好的网络进行预测

        # 二进制转十进制，展开编写，方便计算，提高运行效率
        chaotic_bit_flow = np.zeros((self.student_nstep,1))
        for i in range(self.student_nstep):
            b0 = round(y_predict_ESN[i * 8,0])
            b1 = round(y_predict_ESN[i * 8+1, 0])
            b2 = round(y_predict_ESN[i * 8+2, 0])
            b3 = round(y_predict_ESN[i * 8+3, 0])
            b4 = round(y_predict_ESN[i * 8+4, 0])
            b5 = round(y_predict_ESN[i * 8+5, 0])
            b6 = round(y_predict_ESN[i * 8+6, 0])
            b7 = round(y_predict_ESN[i * 8+7, 0])
            bit = int(1*b0+2*b1+4*b2+8*b3+16*b4+32*b5+64*b6+128*b7)
            chaotic_bit_flow[i,0] = bit

        return chaotic_bit_flow

    # 密钥保存函数
    def SaveKey(self,chaotic_bit_flow,saving_directory,filename='untitled.txt'):
        saving_address = saving_directory+'/'+filename
        np.savetxt(saving_address,chaotic_bit_flow)
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