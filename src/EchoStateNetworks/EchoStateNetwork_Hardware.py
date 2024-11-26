# This code is designed  for realizing echo state network following the work of Yi Zhao' group in HIT, Shenzhen.

import numpy as np
from .Weight_Generation.WeightMatrix import RandomWeightMatrix, NormalizeMatrixElement  # 随机矩阵生成函数
from .Weight_Generation.WeightMatrix_Networkx import Network_initial  # 导入通过networkx生成稀疏连接权重的函数

# 利用scikit-learn优化输出权重， 直接调用以提高运行效率
from .Weight_Optimization.WeightOptimization_sklearn import WeightOptimizing

# reservoir computing
class Analog_ESN():
    def __init__(self, input_dimension=3, output_dimension=3, activation=np.tanh, input_scaling=0.1, bias=0,
                 network_mode='ER', reservoir_dimension=400, reservoir_spectral_radius=1.0, reservoir_density=0.10,
                 transient=1000, random_seed=2048, memory_capacity_config=None, **kwargs):

        rng = np.random.RandomState(random_seed)  # rng - random number generator, 通过random_seed生成随机数列

        self.N = input_dimension              # dimension of the input
        self.M = output_dimension             # dimension of the output
        self.func = activation                # activation function
        self.s_in = input_scaling             # the scaling parameter of the input-to-reservoir connection weight matrix
        self.K = reservoir_dimension          # dimension of the reservoir
        self.rho = reservoir_spectral_radius  # spectral radius of the reservoir
        self.delta = reservoir_density        # density of the reservoir
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
            Network_weight = rng.rand(reservoir_dimension, reservoir_dimension)
            # 生成稀疏矩阵并解压
            W_res_init = (Network_initial(network_mode, Depth=0, random_seed=random_seed,
                                          network_size=reservoir_dimension, density=reservoir_density,
                                          MC_configure=memory_capacity_config)).T
            W_res_init = np.triu(np.multiply(W_res_init,Network_weight))+np.triu(np.multiply(W_res_init,Network_weight)).T
            self.W_res = W_res_init / np.max(np.abs(np.linalg.eigvals(W_res_init)))  # 对储层权重矩阵进行谱半径归一化

    def Training_phase(self, input_train=None, output_train=None, opt_algorithm=0):
        L = input_train.shape[1]  # 训练集长度
        U_state = np.zeros((self.K, L))  # 存放训练阶段所有未激活的储层输入的矩阵
        R_state = np.zeros((self.K, L))  # 存放训练阶段所有储层态的矩阵

        # 更新初始储层态
        u = np.dot(self.rho * self.W_res, R_state[:,0]) + self.s_in * np.dot(self.W_in, input_train[:,0]) + self.b
        U_state[:,0] = u
        R_state[:,0] = self.func(u)
        for i in range(1, L):
            u = np.dot(self.rho * self.W_res, R_state[:,i-1]) + self.s_in * np.dot(self.W_in, input_train[:,i]) + self.b
            U_state[:,i] = u
            R_state[:,i] = self.func(u)

        W_out = WeightOptimizing(output_train[:, self.transient:].T, R_state[:, self.transient:].T,
                                 index=opt_algorithm, k=0.8)
        W_out = W_out.T  # 要转置才符合这套代码的规范

        ESN_output_train = np.dot(W_out, R_state)

        self.W_out = W_out
        self.laststate = R_state[:, -1]
        self.lastinput = output_train[:, -1]

        return (ESN_output_train[:,self.transient:], output_train[:, self.transient:], U_state[:, self.transient:],
                R_state[:, self.transient:], W_out)

    def Predicting_phase(self, Q):
        outputs = np.zeros((self.N,Q))
        U_state = np.zeros((self.K,Q))  # 存放测试阶段所有未激活的储层输入的矩阵
        R_state = np.zeros((self.K,Q))  # 存放测试阶段所有储层态的矩阵

        u = np.dot(self.rho * self.W_res, self.laststate) + self.s_in * np.dot(self.W_in, self.lastinput) + self.b
        U_state[:,0] = u
        R_state[:,0] = self.func(u)
        outputs[:,0] = np.dot(self.W_out,R_state[:,0])
        for i in range(1, Q):
            u = np.dot(self.rho * self.W_res, R_state[:,i-1]) + self.s_in * np.dot(self.W_in, outputs[:,i-1]) + self.b
            U_state[:,i] = u
            R_state[:,i] = self.func(u)
            outputs[:,i] = np.dot(self.W_out,R_state[:,i])

        return outputs, U_state, R_state

    # 输出各级连接权重矩阵
    def ExportingWeight(self):  return self.W_in, self.W_res, self.W_out

if __name__=='__main__':
    pass