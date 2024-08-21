import ASHEN as an
import numpy as np

class Li_ESN:
    '''
    This code is designed to construct an Leaky-integrator Echo State Network (Li-ESN).
    '''

    def __init__(self,x_train: np.ndarray, y_train: np.ndarray, **kwargs):
        '''
        对网络参数进行初始化
        x_train，y_train分别为输入，输出的训练数据集，应为二阶张量，维数格式为 (测试集的长度,输入/输出向量的维数)
        '''

        x_train, y_train = (x_train.T,y_train.T)  # 转置补丁，使得此代码适用于最新式的版本

        x_shape, y_shape = [x_train.shape, y_train.shape]  # 获取输入输出的训练数据的的维度
        # 为了方便表达，我们接下来用简单的大写字母来表示各层的维数
        N, M, L = [x_shape[0], x_shape[1], y_shape[1]]  # 训练集长度，输入维数，输出维数

        # 定义ESN的基本结构：# a=1时，网络结构为最基础的ESN；a<1时，网络结构为Li-ESN
        a = kwargs['leaking_rate'] if 'leaking_rate' in kwargs else 0.5  # ESN的leaking rate
        activation = kwargs['activation'] if 'activation' in kwargs else np.tanh  # 激活函数

        # ESN有三个关键的层：输入，储层，以及输出；输入层与输出层的维数由我们要学习的数据决定，而储层的特性则极大影响了网络的学习能力
        # ESN的输入矩阵以及储层矩阵中的元素，即权重都是随机生成的，在此我们可以定义一开始生成的权重的范围，但是后面权重矩阵都要进行归一化处理，故影响不大
        weight_range = kwargs['weight_element_range'] if 'weight_element_range' in kwargs else (-1, 1)
        # 由于连接权重跟储层态都是随机生成的，为了方便管理，我们在一个网络中只生成一次，每次重新构建网络才会重新生产
        # 首先让我们定义储层权重，储层权重是一个随机的稀疏矩阵，更多细节详见：https://www.sciencedirect.com/science/article/pii/S1574013709000173
        K = kwargs['reservoir_dimension'] if 'reservoir_dimension' in kwargs else 100     # 定义储层维数，也是储层态向量长度
        res_den = kwargs['reservoir_density'] if 'reservoir_density' in kwargs else 0.04  # 储层矩阵密度
        rsr = kwargs['reservoir_spectral_radius'] if 'reservoir_spectral_radius' in kwargs else 0.5  # 储层矩阵的谱半径
        # 那么接下来我们正式定义储层权重矩阵（Reservoir Weight Matrix）
        W_res_init = an.WeightMatrix.GenSparseMatrix((K, K), weight_range, res_den).todense()  # 生成稀疏矩阵并解压
        W_res = rsr*an.WeightMatrix.NormalizeSpectralRadius(W_res_init)  # 对储层权重矩阵进行谱半径归一化，并根据指定谱半径重新scaling

        # 同时，我们一开始需要一个初始化的储层态来启动网络的迭代，在此，默认初始的储层态是一个零向量
        r_init = kwargs['reservoir_state_initial'] if 'reservoir_state_initial' in kwargs else np.zeros((1,K),dtype=float)

        # 同样的，让我们来定义输入层的连接权重矩阵（Input Weight Matrix)
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.3  # 输入的缩放因子
        W_in_init = an.WeightMatrix.RandomWeightMatrix((M, K), weight_range)  # 初始化随机输入权重矩阵
        W_in = s_in*an.WeightMatrix.NormalizeMatrixElement(W_in_init)  # 对输入权重进行矩阵元素归一化，并按照指定缩放因子重新scaling

        # 有时，我们可以加入来自输出的反馈，通过输出反馈权重矩阵（Output Feedback Weight Matrix）连接
        s_fb = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 0.0               # 输出反馈的缩放因子
        W_fb_raw = an.WeightMatrix.RandomWeightMatrix((L, K), weight_range)                      # 生成随机权重矩阵
        W_fb = s_fb*an.WeightMatrix.NormalizeMatrixElement(W_fb_raw)                             # 进行矩阵元素归一化

        # y_train_feedback = np.hstack(([[0,0,0]],y_train[0:N]))
        # print(y_train_feedback)

        # 权重优化函数
        optimization = kwargs['optimization'] if 'optimization' in kwargs else an.LinearRegression.RIDGE  # 默认为岭回归
        self.optimization = optimization  # 转换为实例函数

        # 正则化系数（当使用ridge regression或者LASSO时，需要设置这个系数）
        self.alpha = kwargs['regularization_coefficient'] if 'regularization_coefficient' in kwargs else 1.0  # 默认为1.0

        # 将一些变量转变为实例变量，方便这个class下面的其他函数调用
        self.x_train, self.y_train, self.activation = (x_train,y_train,activation)  # 训练集输入，训练集输出，以及激活函数
        self.N, self.M, self.K, self.L = (N, M, K, L)                               # 训练集长度以及，ESN各层维数
        self.W_in, self.W_res, self.W_fb = (W_in, W_res, W_fb)                      # 各个连接权重
        self.r_init = r_init                                                        # 初始储层态向量
        self.a = a                                                                  # leaking rate
        self.activation = activation                                                # 激活函数

        # （可选项）向我们的网络中加入噪声扰动，提高拟合能力，同时增强鲁棒性（robustness）
        s_noise = kwargs['noise_scaling'] if 'noise_scaling' in kwargs else 0.0     # 噪声的缩放因子
        noise_mode = kwargs['noise_mode'] if 'noise_mode' in kwargs else 'Uniform'   # 噪声模式，默认为均匀分布
        if noise_mode == 'Normal':
            noise_train = np.random.normal(0, 0.2, (N, K))  # 生成形状为NxK的正态分布噪声矩阵
        elif noise_mode == 'Poisson':
            noise_train = np.random.poisson(0,(N,K))  # 生成形状为NxK的泊松分布噪声矩阵
        elif noise_mode == 'Uniform':
            noise_train = np.random.uniform(-0.5, 0.5, (N, K))
        else:
            print('Please enter the correct noise mode:')
            exit()
        self.s_noise = s_noise
        self.noise = s_noise*noise_train

    ####################################################################################################################
    # 训练模块
    # 此函数可以计算训练集输入生成的所有储层态
    # 并利用Tikhonov regularization计算输出权重（https://en.wikipedia.org/wiki/Tikhonov_regularization），并计算网络输出
    def TrainESN(self):
        N, M, K, L = [self.N, self.M, self.K, self.L]  # 训练集长度，输入维数，储层态维数，输出维数
        r_0 = self.r_init  # 初始储层态向量，为零向量
        r_train, v_train = [np.empty((N,K),dtype=float),np.empty((N,M+K),dtype=float)]  # r为训练阶段的所有储层态，v为所有储层态跟输入的串接
        for i in range(N):
            x = self.x_train[i].reshape(1,-1)  # 要将输入转换成二维数组才能进行矩阵运算
            y = self.x_train[i].reshape(1,-1)  # 对于时序回归任务，我们有y[n]=x[n+1]，所以此输出即为下一时刻的输入
            nu = self.noise[i].reshape(1,-1)   # 主动噪声扰动项
            # a = x@self.W_in+r_0@self.W_res+y@self.W_fb+nu
            # print(a[0,:].shape)
            r_1 = (1.0-self.a)*r_0+self.activation(x@self.W_in+r_0@self.W_res+y@self.W_fb+nu)  # 计算储层态
            r_train[i] = r_1.getA()[0]  # 将矩阵转换为数组后取第一个元素（即第一行）以降维
            v_train[i] = np.hstack((r_1,x))[0]  # 将储层态与输入串接（这一步很关键！！！），并将串接向量的值记录在v_train中
            r_0 = r_1

        W_out, threshold = self.optimization(v_train, self.y_train, alpha=self.alpha)  # 利用Tikhonov正则化计算输出连接权重
        y_train_ESN = np.dot(v_train, np.transpose(W_out)) + threshold  # 网络的输出，可用于计算各类统计指标，分析训练结果

        return r_train, y_train_ESN.T, W_out, threshold

    ####################################################################################################################
    # 预测模块
    def Forecasting(self,r_train, W_out, threshold, predicting_step):
        # if 'entering_external_information' in kwargs:
            # r_train, v_train = (kwargs['reservoir_state'], kwargs['concatenated_state'])
            # W_out, threshold = (kwargs['output_connection_weight'],kwargs['threshold'])
        # else:
            # r_train, v_train = self.CalResState()                   # 利用CalResState()函数计算训练集输入生成的所有储层态
            # W_out, threshold, y_train_ESN = self.CalOutputWeight()  # 利用CalOutputWeight()函数计算训练集输入生成的所有储层态

        Q, M, K, L = [predicting_step, self.M, self.K, self.L]        # 要预测的步数，输入维数，储层态维数，输出维数

        noise_predict = self.s_noise*np.random.uniform(-0.5, 0.5, (Q, K))

        x_0 = self.y_train[len(self.y_train)-1].reshape(1,-1)  # 预测阶段的第一个输入是训练阶段的最后一个输出，要注意转换成二阶张量
        y_0 = self.y_train[len(self.y_train)-1].reshape(1,-1)  # 输出反馈项
        r_0 = r_train[len(r_train)-1].reshape(1,-1)            # 水库态的初始值是最后输出的水库态
        # r_predict, v_predict = [np.empty((Q,K),dtype=float), np.empty((Q,M+K),dtype=float)]  # r为预测阶段的所有储层态，v为所有储层态跟输入的串接
        y_predict_ESN = np.empty((Q,L),dtype=float)            # 创建一个二阶张量存放ESN的输出结果
        for i in range(Q):
            x_1 = x_0
            nu = noise_predict[i].reshape(1,-1)                # 主动噪声扰动

            r_1 = (1.0-self.a)*r_0 + self.activation(x_1@self.W_in+r_0@self.W_res+y_0@self.W_fb+nu)
            v_1 = np.hstack((r_1, x_1))
            y_1 = np.dot(v_1, np.transpose(W_out))+threshold

            x_0 = y_1
            y_0 = y_1                                           # 更新输出反馈
            r_0 = r_1

            y_predict_ESN[i] = y_1.getA()[0]

        return y_predict_ESN.T

    ####################################################################################################################
    # 投影分类模块
    def Projecting(self,x_test,r_train,W_out,threshold,projecting_step):
        Q, M, K, L = [projecting_step, self.M, self.K, self.L]  # 要投影的步数，输入维数，储层态维数，输出维数

        x_test = x_test.T
        print(x_test.shape)
        y_test = np.empty((Q,L), dtype=float)  # 网络预测结果
        r_test = np.empty((Q,K), dtype=float)  # r为训练阶段的所有储层态
        v_test = np.empty((Q,M+K), dtype=float)   # v为所有储层态跟输入的串接

        r_0 = r_train[len(r_train) - 1].reshape(1, -1)  # 水库态的初始值是最后输出的水库态
        for i in range(Q):
            x = x_test[i].reshape(1, -1)  # 要将输入转换成二维数组才能进行矩阵运算
            r_1 = (1.0 - self.a) * r_0 + self.activation(x @ self.W_in + r_0 @ self.W_res)  # 计算储层态
            v_1 = np.hstack((r_1, x))  # 将储层态与输入串接（这一步很关键！！！），并将串接向量的值记录在v_train中

            y_1 = np.dot(v_1, W_out.T) + threshold  # 投影得到输出

            r_test[i] = r_1.getA()[0]  # 将矩阵转换为数组后取第一个元素（即第一行）以降维
            v_test[i] = v_1[0]
            y_test[i] = y_1[0]

            r_0 = r_1

        return y_test.T

if __name__=='__main__':
    pass