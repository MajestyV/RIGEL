# In this code, we will define a class ESN4Trajectory, which is designed to predict the trajectory of a single object or
# many-body systems.
# The basic framework is designed based on numpy and pytorch. For the convenience of data conversion between the two
# packages, we will stick to float with 32 bits.
import ASHEN as an
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV,OrthogonalMatchingPursuit

class ESN4Trajectory:
    """
    This code is designed to construct an Echo State Network (ESN, a founding model of Reservoir Computing) for the
    prediction of trajectories of single object or many-body systems.

    The input to this algorithm should be organized in the form of numpy arrays with dimension
    [step number, particle number, degree of freedom]
    算法的输入跟输出的维度应为：[时间步数，粒子数，空间自由度]
    """

    # 对网络参数进行初始化 -------------------------------------------------------------------------------------------------
    def __init__(self,X_train,Y_train,activation=np.tanh,**kwargs):

        X_train, Y_train = (X_train.T, Y_train.T)  # 对输入数据进行转置，适配ASHEN的动态系统模块数据结构

        L_train, N = X_train.shape  # L-train - 训练集长度，N - 粒子数/轨迹数（本质就是特征数）

        # 让我们先定义ESN的基本结构：# a=1时，网络结构为最基础的ESN；a<1时，网络结构为Li-ESN
        a = kwargs['leaky_rate'] if 'leaky_rate' in kwargs else 1.0                    # Li-ESN的leaky rate

        # 定义储层
        K = kwargs['res_dim'] if 'res_dim' in kwargs else 500                          # 定义储层维数，也是储层态向量长度
        rho = kwargs['spec_rad'] if 'spec_rad' in kwargs else 1.0                      # 储层矩阵的谱半径
        theta = kwargs['res_den'] if 'res_den' in kwargs else 0.1                      # 储层矩阵密度
        # 那么接下来我们正式定义储层权重矩阵（Reservoir Weight Matrix）
        W_res_init = an.WeightMatrix.GenSparseMatrix((K, K), (-1,1), theta).todense()  # 生成稀疏矩阵并解压
        W_res = rho * an.WeightMatrix.NormalizeSpectralRadius(W_res_init)  # 对储层权重矩阵进行谱半径归一化，并根据指定谱半径重新scaling
        # 同时，我们一开始需要一个初始化的储层态来启动网络的迭代，在此，默认初始的储层态是一个零向量
        r_init = kwargs['init_res_state'] if 'init_res_state' in kwargs else np.zeros((1,K), dtype=np.float32)
        # 应注意，为了方便各种态在numpy和pytorch之间的转换，我们将所有的矩阵都定义为float32类型

        # 接下来定义输入层的连接权重矩阵（Input Weight Matrix)
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.1  # 输入的缩放因子
        W_in_init = an.WeightMatrix.RandomWeightMatrix((N,K), (-1, 1))  # 初始化随机输入权重矩阵
        W_in = s_in * an.WeightMatrix.NormalizeMatrixElement(W_in_init)  # 对输入权重进行矩阵元素归一化，并按照指定缩放因子重新scaling

        # 批量转化变量转变为实例变量，方便这个class下面的其他函数调用
        self.X_train, self.Y_train, self.activation = (X_train,Y_train,activation)  # 训练集输入，训练集输出，以及激活函数
        self.N, self.K, self.L_train = (N, K, L_train)                              # 训练集长度以及ESN各层维数
        self.W_in, self.W_res = (W_in, W_res)                                       # 各个连接权重
        self.r_init = r_init                                                        # 初始储层态
        self.a = a                                                                  # Leaky rate

    # 回声状态储层，此函数可以根据给定的训练集轨迹产生对应的储层轨迹
    def EchoStateReservoir(self):
        X_train, Y_train = (self.X_train,self.Y_train)             # 训练集
        N, K, L_train = (self.N, self.K, self.L_train)             # 训练集长度，输入维数，储层态维数，输出维数
        W_in, W_res = (self.W_in,self.W_res)                       # 随机生成的输入权重和储层连接权重
        a = self.a                                                 # Leaky rate

        # 创建几个空矩阵用于存放数据
        R_train = np.zeros((L_train,K),dtype=np.float32)    # R_train为训练阶段的所有储层态
        V_train = np.zeros((L_train,K+N),dtype=np.float32)  # V_train为所有储层态跟输入的串接，也就是所有的修正储层态

        r_0 = self.r_init  # 初始储层态向量，为零向量（同时代表迭代时的r_n储层态）
        r_1 = self.r_init  # 初始化一个储层态作为迭代时的r_n+1储层态
        for i in range(L_train):
            x = X_train[i].reshape(1,-1)  # 将一维数组reshape成二维数组，方便进行矩阵运算
            r_1 = (1.0-a)*r_0+self.activation(np.dot(x,W_in)+np.dot(r_0,W_res))
            v_1 = np.concatenate((r_1,x),axis=1)  # 对于轨迹预测问题，将储层态和输入串接在一起，形成修正储层态，可以有效修正轨迹

            r_0 = r_1  # 迭代更新储层态（关键代码）

            # 保存数据
            R_train[i] = r_1[0]
            V_train[i] = v_1[0]

        r_state_present = r_1  # 输出此时的储层态

        return V_train, R_train, r_state_present

    # Reservoir encoder模块 (储层编码器)
    def Encoder(self):
        return

    # Reservoir iterator模块 (储层迭代器)
    def Iterator(self,output_projection):
        N, K = (self.N, self.K)
        W_in, W_res = (self.W_in, self.W_res)
        a = self.a  # Leaky rate

        # 创建几个空矩阵用于存放数据
        R_test = np.zeros((L_test, K), dtype=np.float32)  # R_train为训练阶段的所有储层态
        V_test = np.zeros((L_test, K + N), dtype=np.float32)  # V_train为所有储层态跟输入的串接
        ESN_test = np.zeros((L_test, N), dtype=np.float32)  # Y_train为预测阶段的输出

        x = self.Y_train[-1].reshape(1, -1)  # 预测阶段的初始输入为训练阶段的最后一个输出
        r_0 = r_test_init  # 预测阶段的初始储层态向量
        r_1 = self.r_init  # 初始化一个储层态作为迭代时的r_n+1储层态
        for i in range(L_test):
            r_1 = (1.0 - a) * r_0 + self.activation(np.dot(x, W_in) + np.dot(r_0, W_res))
            v_1 = np.concatenate((r_1, x), axis=1)

            y = np.dot(v_1, W_out.T) + threshold  # 根据储层态和输出权重矩阵计算输出

            # 迭代演化
            x = y  # 此时刻的输出将成为下一时刻的输入（迭代演化的核心代码）
            r_0 = r_1

            # 记录数据
            R_test[i] = r_1[0]
            V_test[i] = v_1[0]
            ESN_test[i] = y[0]

        r_state_present = r_1  # 输出此时的储层态

        return ESN_test.T, V_test.T, R_test.T, r_state_present  # 转置以适配ASHEN的数据结构

if __name__=='__main__':
    pass