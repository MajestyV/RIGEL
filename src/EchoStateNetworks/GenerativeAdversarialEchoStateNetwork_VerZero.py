# This code is designed  for realizing echo state network following the work of Yi Zhao' group in HIT, Shenzhen.

import numpy as np
# 直接调用包内函数，提高运行效率
from ..Weight_Generation.WeightMatrix import RandomWeightMatrix, NormalizeMatrixElement  # 随机矩阵生成函数
from ..Weight_Generation.WeightMatrix_Networkx import Network_initial  # 导入通过networkx生成稀疏连接权重的函数
from ..Weight_Optimization.WeightOptimization_sklearn import WeightOptimizing  # 利用scikit-learn优化输出权重

class GAESN():
    '''
    This code is designed for implementing the echo state network (ESN, a founding model of reservoir computing) model.
    '''
    def __init__(self, input_dimension: int, output_dimension: int, activation: callable=np.sin,
                 input_scaling: float=0.1, bias: float=0, network_mode: str='ER',
                 res_dim: int=400, spec_rad: float=1.0, res_den: float=0.10, transient: int=0,
                 fixed_random_number: bool=True, random_seed: int=42, memory_capacity_config=None, **kwargs):

        # 由于模型涉及一系列的随机数生成，我们可以通过设置一个固定的随机种子来保证每次运行的结果一致
        if fixed_random_number:
            rng = np.random.RandomState(random_seed)  # rng - random number generator, 通过random_seed生成随机数列
        else:
            rng = np.random.RandomState()             # 如果不固定随机数，那么就使用系统时间作为随机种子

        #input_dimension, output_dimension = (X_train.shape[0], Y_train.shape[0])  # 读取输入输出维度

        #self.X_train = X_train                # 训练集输入
        #self.Y_train = Y_train                # 训练集输出
        #self.L_train = X_train.shape[1]       # 训练集长度
        self.N = input_dimension              # dimension of the input
        self.M = output_dimension             # dimension of the output
        self.func = activation                # activation function
        self.s_in = input_scaling             # the scaling parameter of the input-to-reservoir connection weight matrix
        self.K = res_dim                      # dimension of the reservoir
        self.rho = spec_rad                   # spectral radius of the reservoir
        self.delta = res_den                  # density of the reservoir
        self.b = bias                         # the bias term (to the reservoir)
        self.transient = transient            # the number of reservoir state data to be deleted

        # ESN有三个关键的层：输入，储层，以及输出；输入层与输出层的维数由我们要学习的数据决定，而储层的特性则极大影响了网络的学习能力
        # ESN的输入矩阵以及储层矩阵中的元素，即权重都是随机生成的，在此我们可以定义一开始生成的权重的范围，但是后面权重矩阵都要进行归一化处理，故影响不大
        weight_range = kwargs['weight_element_range'] if 'weight_element_range' in kwargs else (-1, 1)
        # 输入连接权重
        if 'input_connection_weight' in kwargs:
            self.W_in = kwargs['input_connection_weight']
        else:
            # 初始化随机输入权重矩阵
            W_in_init = RandomWeightMatrix((self.K, self.N), weight_range,
                                           random_seed=random_seed, lock=True)
            self.W_in = NormalizeMatrixElement(W_in_init)  # 对输入权重进行矩阵元素归一化
        # 储层连接权重
        if 'reservoir_connection_weight' in kwargs:
            self.W_res = kwargs['reservoir_connection_weight']
        else:
            Network_weight = rng.rand(res_dim, res_dim)
            # 生成稀疏矩阵并解压
            W_res_init = (Network_initial(network_mode, Depth=0, random_seed=random_seed,
                                          network_size=res_dim, density=res_dim,
                                          MC_configure=memory_capacity_config)).T
            W_res_init = np.triu(np.multiply(W_res_init,Network_weight))+np.triu(np.multiply(W_res_init,Network_weight)).T
            self.W_res = W_res_init / np.max(np.abs(np.linalg.eigvals(W_res_init)))  # 对储层权重矩阵进行谱半径归一化

    # ------------------------------------------------------------------------------------------------------------------
    def Generator(self, L_train: int, L_test: int):
        '''
        This function is designed for training the echo state network model.
        '''
        L = L_train+L_test

        T = np.arange(0, L, 1).reshape(1,-1)  # 时序信号

        self.R = np.zeros((self.K, L), dtype=float)  # 初始化一个矩阵用于存放储层态

        # 更新初始储层态
        self.R[:, 0] = self.func(np.dot(self.rho * self.W_res, self.R[:, 0])
                                       + self.s_in * np.dot(self.W_in, T[:, 0]) + self.b)
        for i in range(L):
            self.R[:, i] = self.func(np.dot(self.rho * self.W_res, self.R[:, i - 1])
                                           + self.s_in * np.dot(self.W_in, T[:, i]) + self.b)

        self.R_train = self.R[:, :L_train]
        self.R_test = self.R[:, L_train:]

        return T[:, :L_train], T[:, L_train:]

    def Projector(self, Y_train: int, opt_algorithm: int=0):
        '''

        '''
        # print(self.Y_train[:, self.transient:].T.shape,R_state[:, self.transient:].T.shape)  # For debugging
        W_out = WeightOptimizing(Y_train[:, self.transient:].T, self.R_train[:, self.transient:].T,
                                 index=opt_algorithm, k=0.8)
        W_out = W_out.T  # 要转置才符合这套代码的规范

        self.ESN_train = np.dot(W_out, self.R_train)

        # 将一些变量转换为实例变量，方便全局调用
        self.W_out = W_out
        self.r_state_trained = self.R_train[:, -1]
        self.init_input_test = Y_train[:, -1]  # 训练集的最后一个输出用于预测的初始输入

        return self.ESN_train[:,self.transient:]
        # self.Y_train[:, self.transient:], R_state, W_out

    def Predictor(self, ):
        '''

        '''
        self.ESN_test = np.dot(self.W_out, self.R_test)

        return self.ESN_test



    # ------------------------------------------------------------------------------------------------------------------
    def Predicting(self, L_test: int):
        '''
        This function is designed for predicting the output of the echo state network model.
        '''
        self.ESN_test = np.zeros((self.N,L_test))
        self.R_test = np.zeros((self.K,L_test))

        self.R_test[:,0] = self.func((np.dot(self.rho * self.W_res, self.r_state_trained)
                                     + self.s_in * np.dot(self.W_in, self.init_input_test) + self.b))
        self.ESN_test[:,0] = np.dot(self.W_out,self.R_test[:,0])
        for i in range(1, L_test):
            self.R_test[:,i] = self.func(np.dot(self.rho * self.W_res, self.R_test[:,i-1])
                                    + self.s_in * np.dot(self.W_in, self.ESN_test[:,i-1]) + self.b)

            self.ESN_test[:,i] = np.dot(self.W_out,self.R_test[:,i])

        self.r_state_latest = self.R_test[:, -1]

        return self.ESN_test

    # ------------------------------------------------------------------------------------------------------------------
    #def Projecting_phase(self, x_test):
        #'''
        #This function is designed for projecting the output of the echo state network model.
        #'''
        #Q = x_test.shape[1]

        #R_state = np.zeros((self.K, Q))

        #R_state[:, 0] = self.func((np.dot(self.rho * self.W_res, self.laststate)
                                   #+ self.s_in * np.dot(self.W_in, self.lastinput) + self.b))
        #for i in range(1, Q):
            #R_state[:,i] = self.func(np.dot(self.rho * self.W_res, R_state[:,i-1])
                                    #+ self.s_in * np.dot(self.W_in, x_test[:,i]) + self.b)

        #outputs = np.dot(self.W_out, R_state)

        #return outputs

    #################################################### 辅助分析模块 ####################################################

    def Get_Weight(self):
        '''
        输出各级连接权重矩阵
        '''
        return (self.W_in, self.W_res, self.W_out)

    def Get_ESN_states(self, period: str):
        '''
        获取ERA在运行过程中的状态数据
        '''
        ESN_states_dict = {'train': (self.R_train, self.ESN_train, self.r_state_trained),  # 训练阶段
                           'test': (self.R_test, self.ESN_test, self.r_state_latest),  # 预测阶段
                           'total': (np.vstack((self.R_train, self.R_test)),  # 整个过程
                                     np.vstack((self.ESN_train, self.ESN_test)),
                                     np.array([self.r_state_trained, self.r_state_latest]))}
        return ESN_states_dict[period]

if __name__=='__main__':
    pass