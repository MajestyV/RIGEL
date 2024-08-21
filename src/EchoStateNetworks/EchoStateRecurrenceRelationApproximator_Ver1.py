# This code is designed for constructing Echo state Recurrence relation Approximator (ERA).
# This is a development version.

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import ASHEN as an

import torch
import torch.nn as nn
from tqdm import tqdm

class ERA:
    '''
    Echo state Recurrence relation Approximator (ERA)

    This class is the development version of ERA model, which is designed for approximate and predict the trajectories
    of dynamical systems with discretized dynamics.
    本类是ERA模型的开发版本，用于拟合和预测离散动力学系统的轨迹。

    The input to this algorithm should be organized in the form of numpy arrays with dimension
    [step number, particle number, degree of freedom]
    算法的输入跟输出的维度应为：[时间步数]
    '''

    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, X_validation: np.ndarray, Y_validation: np.ndarray,
                 random_seed: int = 10, fixed_random_number: bool = True, **kwargs):
        '''
        对网络参数进行初始化
        '''

        # 由于模型涉及一系列的随机数生成，我们可以通过设置一个固定的随机种子来保证每次运行的结果一致
        if fixed_random_number:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        else:
            pass

        # 对输入数据进行转置，适配ASHEN的动态系统模块数据结构，顺便转变为实例变量
        self.X_train, self.Y_train, self.X_valid, self.Y_valid = (X_train.T, Y_train.T, X_validation.T, Y_validation.T)
        # 提取数据结构参数
        self.L_train, self.N = self.X_train.shape  # L-train - 训练集长度，N - 粒子数/轨迹数（本质就是特征数）
        self.L_valid = self.X_valid.shape[0]  # 验证集长度

        # 定义网络超参： a=1时，网络结构为最基础的ESN；a<1时，网络结构为Li-ESN
        self.a = kwargs['leaky_rate'] if 'leaky_rate' in kwargs else 1.0  # Li-ESN的leaky rate
        self.activation = kwargs['activation'] if 'activation' in kwargs else np.tanh  # 激活函数

        # 定义储层
        self.K = kwargs['res_dim'] if 'res_dim' in kwargs else 500  # 定义储层维数，也是储层态向量长度
        rho = kwargs['spec_rad'] if 'spec_rad' in kwargs else 1.0  # 储层矩阵的谱半径
        theta = kwargs['res_den'] if 'res_den' in kwargs else 0.1  # 储层矩阵密度
        # 那么接下来我们正式定义储层权重矩阵（Reservoir Weight Matrix）
        W_res_init = an.WeightMatrix.GenSparseMatrix((self.K, self.K), (-1, 1), theta).todense()  # 生成稀疏矩阵并解压
        self.W_res = rho * an.WeightMatrix.NormalizeSpectralRadius(W_res_init)  # 对储层权重矩阵进行谱半径归一化，并根据指定谱半径重新scaling
        # 同时，我们一开始需要一个初始化的储层态来启动网络的迭代，在此，默认初始的储层态是一个零向量
        self.r_init = kwargs['init_res_state'] if 'init_res_state' in kwargs else np.zeros((1, self.K),
                                                                                           dtype=np.float32)
        # 应注意，为了方便各种态在numpy和pytorch之间的转换，我们将所有的矩阵都定义为float32类型

        # 接下来定义输入层的连接权重矩阵（Input Weight Matrix)
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.1  # 输入的缩放因子
        W_in_init = an.WeightMatrix.RandomWeightMatrix((self.N, self.K), (-1, 1))  # 初始化随机输入权重矩阵
        self.W_in = s_in * an.WeightMatrix.NormalizeMatrixElement(W_in_init)  # 对输入权重进行矩阵元素归一化，并按照指定缩放因子重新scaling

        # 定义输出连接矩阵
        self.W_out = torch.zeros((self.K+self.N,self.N),requires_grad=True)  # 输出连接权重矩阵

    # 定义输出投影层 -----------------------------------------------------------------------------------------------------
    def OutputProjection(self, v: torch.Tensor):
        '''
        输出层的投影函数，此函数可以将储层态投影到输出空间
        '''
        return torch.mm(v,self.W_out)  # 输出层的投影函数

    # 训练模型 ----------------------------------------------------------------------------------------------------------
    def EchoStateEncoder(self):
        '''
        Echo State Reservoir Encoder模块 (回声状态储层编码器)，此函数可以根据给定的训练集轨迹产生对应的储层轨迹
        '''
        X_train, Y_train = (self.X_train, self.Y_train)  # 训练集
        N, K, L_train = (self.N, self.K, self.L_train)  # 训练集长度，输入维数，储层态维数，输出维数
        W_in, W_res = (self.W_in, self.W_res)  # 随机生成的输入权重和储层连接权重
        a = self.a  # Leaky rate

        # 创建几个空矩阵用于存放数据
        self.R_train = np.zeros((L_train, K), dtype=np.float32)  # R_train为训练阶段的所有储层态
        self.V_train = np.zeros((L_train, K + N), dtype=np.float32)  # V_train为所有储层态跟输入的串接，也就是所有的修正储层态

        r_0 = self.r_init  # 初始储层态向量，为零向量（同时代表迭代时的r_n储层态）
        r_1 = self.r_init  # 初始化一个储层态作为迭代时的r_n+1储层态
        for i in range(L_train):
            x = X_train[i].reshape(1, -1)  # 将一维数组reshape成二维数组，方便进行矩阵运算
            r_1 = (1.0 - a) * r_0 + self.activation(np.dot(x, W_in) + np.dot(r_0, W_res))
            v_1 = np.concatenate((r_1, x), axis=1)  # 对于轨迹预测问题，将储层态和输入串接在一起，形成修正储层态，可以有效修正轨迹

            r_0 = r_1  # 迭代更新储层态（关键代码）

            # 保存数据
            self.R_train[i] = r_1[0]
            self.V_train[i] = v_1[0]

        self.r_state_encoded = r_1  # 输出此时的储层态

        return  # 返回值为空，所有的重要数据都以转换为实例变量，可以利用分析模块按需输出

    def TrainEncoder(self, bias: bool = False, optimizer_mode: str = 'Adam', lr: float = 0.001,
                     loss_function: nn.Module = an.API_for_PyTorch.Loss_RMSE(), nepochs_train: int = 2000):
        '''
        Echo State Reservoir Encoder训练模块 (回声状态储层编码器训练模块)，此函数可以根据给定的训练集轨迹产生对应的储层轨迹
        '''
        input_dim, output_dim = (self.V_train.shape[-1], self.Y_train.shape[-1])  # 获取输入和输出的维数
        # output_projection = an.API_for_PyTorch.LinearProjection(input_dim, output_dim, bias=bias)  # 导入线性层模型
        # 设置优化器
        optimizer = an.API_for_PyTorch.PyTorch_optimizer([self.W_out], optimizer=optimizer_mode, learning_rate=lr)

        # 开始训练模型
        self.loss_record_encoder = np.zeros((nepochs_train, 2), dtype=np.float32)  # 创建一个零数组用于记录encoder训练的loss
        # 将numpy数组转换为tensor
        input = torch.from_numpy(self.V_train)
        target = torch.from_numpy(self.Y_train)
        for epoch in tqdm(range(nepochs_train)):
            optimizer.zero_grad()  # 梯度要清零每一次迭代
            output: torch.Tensor = self.OutputProjection(input)  # 前向传播
            loss = loss_function(target, output)  # 计算loss
            loss.backward()  # 计算梯度，并返向传播

            # ----------------------------------------------------------------------------------------------------------
            # print(output)
            # print('\tgrad:', output.grad)  # 查看梯度
            # ----------------------------------------------------------------------------------------------------------

            optimizer.step()  # 更新权重参数

            self.loss_record_encoder[epoch] = (epoch, loss.item())  # 记录此epoch的loss
            epoch += 1  # 计数器加一

        self.ERA_train = self.OutputProjection(torch.from_numpy(self.V_train)).detach().numpy()  # 训练集的ESN输出

        self.nepochs_train = nepochs_train  # 将训练过程的epoch数转化为实例变量

        return self.ERA_train.T  # 对于网络输出，需要转置以适配ASHEN的数据结构

    # 迭代验证 ----------------------------------------------------------------------------------------------------------

    def EchoStateIterator(self):
        '''
        Echo State Reservoir Iterator模块 (回声状态储层迭代器)，此函数可以通过给定的输出连接output_projection对储层进行迭代
        '''
        N, K = (self.N, self.K)
        W_in, W_res = (self.W_in, self.W_res)
        a = self.a  # Leaky rate

        # 创建几个空矩阵用于存放数据
        R_valid = np.zeros((self.L_valid, K), dtype=np.float32)  # R_train为训练阶段的所有储层态
        V_valid = np.zeros((self.L_valid, K + N), dtype=np.float32)  # V_train为所有储层态跟输入的串接
        ERA_valid = np.zeros((self.L_valid, N), dtype=np.float32)  # Y_train为预测阶段的输出

        x = self.Y_train[-1].reshape(1, -1)  # 预测阶段的初始输入为训练阶段的最后一个输出
        r_0 = self.r_state_encoded  # 预测阶段的初始储层态向量
        r_1 = self.r_init  # 初始化一个储层态作为迭代时的r_n+1储层态
        for i in range(self.L_valid):
            r_1 = (1.0 - a) * r_0 + self.activation(np.dot(x, W_in) + np.dot(r_0, W_res))
            v_1 = np.concatenate((r_1, x), axis=1)

            v_1 = torch.from_numpy(v_1).float()  # 转换为pytorch的tensor，同时通过.float()确保其为32位浮点数张量，防止计算出错
            y = self.OutputProjection(v_1)
            y = y.detach().numpy()  # 转换为numpy的array

            # 迭代演化
            x = y  # 此时刻的输出将成为下一时刻的输入（迭代演化的核心代码）
            r_0 = r_1

            # 记录数据
            R_valid[i] = r_1[0]
            V_valid[i] = v_1[0]
            ERA_valid[i] = y[0]

        r_state_iterated = r_1  # 输出此时的储层态

        return ERA_valid, V_valid, R_valid, r_state_iterated

    def TrainIterator(self, optimizer_mode: str = 'Adam', lr: float = 0.001,
                      loss_function: nn.Module = an.API_for_PyTorch.Loss_RMSE(), nepochs_valid: int = 100):
        '''
        Echo State Reservoir Iterator训练模块 (回声状态储层迭代器训练模块)，此函数可以通过给定的输出连接output_projection对储层进行迭代
        '''
        optimizer = an.API_for_PyTorch.PyTorch_optimizer([self.W_out], optimizer=optimizer_mode, learning_rate=lr)

        # 开始训练模型
        self.loss_record_iterator = np.zeros((nepochs_valid, 2), dtype=np.float32)  # 创建一个零数组用于记录loss
        # 将numpy数组转换为tensor
        target = torch.from_numpy(self.Y_valid)  # 验证集的输出（The ground truth output of the validation set）
        for epoch in tqdm(range(nepochs_valid)):
            # optimizer.zero_grad()  # 梯度要清零每一次迭代
            # 通过迭代器产生验证集的储层态输出
            ERA_valid, V_valid, R_valid, r_state_iterated = self.EchoStateIterator()
            # 前向传播，此时的ESN_valid为numpy数组，需将其转换为pytorch的tensor，而且要加入gradient才能利用pytorch的自动微分功能进行反向传播优化
            output: torch.Tensor = torch.tensor(ERA_valid, requires_grad=True, dtype=torch.float32)  # 前向传播
            # output: torch.Tensor = torch.from_numpy(ERA_valid, requires_grad=True).float()  # 前向传播

            optimizer.zero_grad()  # 梯度要清零每一次迭代

            loss = loss_function(target, output)  # 计算loss
            loss.backward()  # 计算梯度，并返向传播

            # ----------------------------------------------------------------------------------------------------------
            # print(output)
            # print('\tgrad:', output.grad)  # 查看梯度
            # ----------------------------------------------------------------------------------------------------------

            optimizer.step()  # 更新权重参数

            self.loss_record_iterator[epoch] = (self.nepochs_train+epoch, loss.item())  # 记录此epoch的loss
            epoch += 1  # 计数器加一

        # 利用训练好的网络进行预测，同时将所有输出都转化为实例变量，方便后续分析时按需输出
        self.ERA_valid, self.V_valid, self.R_valid, self.r_state_iterated = self.EchoStateIterator()

        return self.ERA_valid.T  # 对于网络输出，需要转置以适配ASHEN的数据结构

    # 外推预测 ----------------------------------------------------------------------------------------------------------

    def EchoStatePredictor(self, L_test: int):
        '''
        Echo State Reservoir Predictor模块 (回声状态储层预测器)，此函数可以通过给定的输出连接output_projection对储层进行迭代
        '''
        N, K = (self.N, self.K)
        W_in, W_res = (self.W_in, self.W_res)
        a = self.a  # Leaky rate

        # 创建几个空矩阵用于存放数据
        self.R_test = np.zeros((L_test, K), dtype=np.float32)  # R_train为训练阶段的所有储层态
        self.V_test = np.zeros((L_test, K + N), dtype=np.float32)  # V_train为所有储层态跟输入的串接
        self.ERA_test = np.zeros((L_test, N), dtype=np.float32)  # Y_train为预测阶段的输出

        x = self.Y_valid[-1].reshape(1, -1)  # 预测阶段的初始输入为验证阶段的最后一个输出
        r_0 = self.r_state_iterated  # 预测阶段的初始储层态向量是验证阶段的最后一个储层态
        r_1 = self.r_init  # 初始化一个储层态作为迭代时的r_n+1储层态
        for i in range(L_test):
            r_1 = (1.0 - a) * r_0 + self.activation(np.dot(x, W_in) + np.dot(r_0, W_res))
            v_1 = np.concatenate((r_1, x), axis=1)

            v_1 = torch.from_numpy(v_1).float()  # 转换为pytorch的tensor，同时通过.float()确保其为32位浮点数张量，防止计算出错
            y = self.OutputProjection(v_1)
            y = y.detach().numpy()  # 转换为numpy的array

            # 迭代演化
            x = y  # 此时刻的输出将成为下一时刻的输入（迭代演化的核心代码）
            r_0 = r_1

            # 记录数据
            self.R_test[i] = r_1[0]
            self.V_test[i] = v_1[0]
            self.ERA_test[i] = y[0]

        self.r_state_latest = r_1  # 最新的的储层态

        return self.ERA_test.T  # 对于网络输出，需要转置以适配ASHEN的数据结构

    #################################################### 辅助分析模块 ####################################################

    def GetLossRecord(self):
        '''
        获取训练过程的loss记录
        '''
        loss_record = np.vstack((self.loss_record_encoder, self.loss_record_iterator))  # 将训练和迭代的loss记录合并
        epoch, loss = (loss_record[:, 0], loss_record[:, 1])  # 整理loss记录
        return epoch, loss

    def Get_ERA_states(self, period: str):
        '''
        获取ERA在运行过程中的状态数据
        '''
        ERA_states_dict = {'train': (self.R_train, self.V_train, self.ERA_train, self.r_state_encoded),  # 训练阶段
                           'valid': (self.R_valid, self.V_valid, self.ERA_valid, self.r_state_iterated),  # 验证阶段
                           'test': (self.R_test, self.V_test, self.ERA_test, self.r_state_latest),  # 预测阶段
                           'total': (np.vstack((self.R_train, self.R_train, self.R_test)),  # 整个过程
                                     np.vstack((self.V_train, self.V_valid, self.V_test)),
                                     np.vstack((self.ERA_train, self.ERA_valid, self.ERA_test)),
                                     np.array([self.r_state_encoded, self.r_state_iterated, self.r_state_latest]))}
        return ERA_states_dict[period]