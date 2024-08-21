# In this code, we will define a class ESN4Trajectory, which is designed to predict the trajectory of a single object or
# many-body systems.
# The basic framework is designed based on numpy and pytorch. For the convenience of data conversion between the two
# packages, we will stick to float with 32 bits.
import torch

import ASHEN as an
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.linear_model import LassoCV,RidgeCV,Ridge,ElasticNetCV,orthogonal_mp,OrthogonalMatchingPursuit

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
        L_train, D, N = X_train.shape  # L-train - 训练集长度，D - 空间自由度, N - 粒子数/轨迹数（本质就是特征数）

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
        r_init = kwargs['init_res_state'] if 'init_res_state' in kwargs else np.zeros((D, K), dtype=np.float32)
        # 应注意，为了方便各种态在numpy和pytorch之间的转换，我们将所有的矩阵都定义为float32类型

        # 接下来定义输入层的连接权重矩阵（Input Weight Matrix)
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.1  # 输入的缩放因子
        W_in_init = an.WeightMatrix.RandomWeightMatrix((N, K), (-1, 1))  # 初始化随机输入权重矩阵
        W_in = s_in * an.WeightMatrix.NormalizeMatrixElement(W_in_init)  # 对输入权重进行矩阵元素归一化，并按照指定缩放因子重新scaling

        # 批量转化变量转变为实例变量，方便这个class下面的其他函数调用
        self.X_train, self.Y_train, self.activation = (X_train,Y_train,activation)  # 训练集输入，训练集输出，以及激活函数
        self.a = a                                                                  # Leaky rate
        self.N, self.D, self.K, self.L_train = (N, D, K, L_train)                   # 训练集长度以及，ESN各层维数
        self.W_in, self.W_res = (W_in, W_res)                                       # 各个连接权重
        self.r_init = r_init

    # 回声状态储层，此函数可以根据给定的训练集轨迹产生对应的储层轨迹
    def EchoStateReservoir(self):
        X_train, Y_train = (self.X_train,self.Y_train)             # 训练集
        N, D, K, L_train = (self.N, self.D, self.K, self.L_train)  # 训练集长度，输入维数，储层态维数，输出维数
        W_in, W_res = (self.W_in,self.W_res)                       # 随机生成的输入权重和储层连接权重
        a = self.a                                                 # Leaky rate

        # 创建几个空矩阵用于存放数据
        # R_train = np.zeros((L_train,D,K),dtype=np.float32)  # R_train为训练阶段的所有储层态
        V_train = np.zeros((L_train,D,K+N),dtype=np.float32)  # V_train为所有储层态跟输入的串接，也就是所有的修正储层态

        r_0 = self.r_init  # 初始储层态向量，为零向量（同时代表迭代时的r_n储层态）
        r_1 = self.r_init  # 迭代时的r_n+1储层态
        for i in range(L_train):
            x = X_train[i]
            r_1 = (1.0-a)*r_0+self.activation(np.dot(x,W_in)+np.dot(r_0,W_res))
            v_1 = np.concatenate((r_1,x),axis=1)  # 对于轨迹预测问题，将储层态和输入串接在一起，形成修正储层态，可以有效修正轨迹

            r_0 = r_1  # 更新储层态

            # R_train[i] = r_1
            V_train[i] = v_1

        r_test_init = r_1  # 训练阶段最后的储层态为外推预测阶段的储层态初始值

        return V_train, r_test_init

    # 训练及预测模块（Scikit-Learn）---------------------------------------------------------------------------------------
    # 此函数可以计算训练集输入生成的所有储层态
    # 并利用Tikhonov regularization计算输出权重（https://en.wikipedia.org/wiki/Tikhonov_regularization），并计算网络输出
    #def Train_sklearn(self,V_train,opt_algorithm=an.LinearRegression.RIDGE,alpha=0.01):
        #W_out, threshold = opt_algorithm(V_train, self.Y_train, alpha=alpha)  # 利用Tikhonov正则化计算输出连接权重
        #y_train_ESN = np.dot(v_train, np.transpose(W_out)) + threshol
        #d  # 网络的输出，可用于计算各类统计指标，分析训练结果
        #return

    # 外推预测模块 -------------------------------------------------------------------------------------------------------
    # 此函数可以根据训练集的储层态和输出权重矩阵，预测未来的轨迹
    def Predict(self,output_projection,r_test_init,L_test):
        N, D, K = (self.N, self.D, self.K)
        W_in, W_res = (self.W_in, self.W_res)  # 随机生成的输入权重和储层连接权重
        a = self.a  # Leaky rate

        # 创建几个空矩阵用于存放数据
        # R_train = np.zeros((L_test, D, K), dtype=np.float32)  # R_train为训练阶段的所有储层态
        V_test = np.zeros((L_test, D, K + N), dtype=np.float32)  # V_train为所有储层态跟输入的串接
        Y_test = np.zeros((L_test, D, N), dtype=np.float32)  # Y_train为预测阶段的输出

        x = self.Y_train[-1]  # 预测阶段的初始输入为训练阶段的最后一个输出
        r_0 = r_test_init  # 预测阶段的初始储层态向量
        for i in range(L_test):
            r_1 = (1.0-a)*r_0+self.activation(np.dot(x,W_in)+np.dot(r_0,W_res))
            v_1 = np.concatenate((r_1,x),axis=1)

            v_1 = torch.from_numpy(v_1).float()  # 转换为pytorch的tensor，同时通过.float()确保其为32位浮点数张量，防止计算出错
            # print(v_1.shape)
            y = output_projection(v_1)
            y = y.detach().numpy()  # 转换为numpy的array

            x = y  # 迭代演化的核心代码
            r_0 = r_1  # 这个很关键！Pytorch版本也没有！

            # 记录数据
            V_test[i] = v_1
            Y_test[i] = y

        return Y_test, V_test

if __name__=='__main__':
    def WeightOptimization(self, Y_train, V_train):
        N, D, K, L_train = (self.N, self.D, self.K, self.L_train)  # 质点数，空间自由度，储层维数，训练集长度
        W_out = np.zeros((N,K+N),dtype=float)

        alphas = 10 ** np.linspace(-4, 2, 7)
        if index == 1:
            base_cv = LassoCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        elif index == 2:
            base_cv = RidgeCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        elif index == 3:
            base_cv = ElasticNetCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)

        elif index == 4:
            anova_filter = SelectKBest(f_regression,
                                           k=int(R_state.shape[1] * k))  # int(self.n_reservoir*0.8 ))#k_number)

        elif index == 5:
            base = linear_model.LinearRegression(fit_intercept=True)
            anova_filter = RFECV(base)
        elif index == 6:
            base_cv = OrthogonalMatchingPursuit(fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)



        classifier = Pipeline([
                ('feature_selection', anova_filter),
                ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
            ])
        for X_i in range(train_data.shape[1]):
            classifier.fit(R_state, train_data[:, X_i])
            W_out[classifier.named_steps['feature_selection'].get_support(), X_i] = classifier.named_steps[
                    'Linearregression'].coef_
        return W_out