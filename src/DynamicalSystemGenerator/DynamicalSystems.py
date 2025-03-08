import numpy as np

########################################################################################################################
#注意事项:
#1. 利用这个函数包模拟动态系统(Mackey-Glass系统除外）时, 应注意步长要设置合适, 如: 0.0001~0.01之间,
#   太小的步长会导致数据量的暴涨, 太大的步长会使系统发散, 无法捕捉真实的动态
#
########################################################################################################################

########################################################################################################################
# 1D dynamical systems

# Logistic map
def Logistic_map(x0=0.5,r=3.54,num_step=5000):
    time = np.zeros((num_step+1, 1))  # 创建空矩阵以存放时序信号
    x = np.zeros((num_step+1,1))      # 创建一个形状为(num_step+1)X1的矩阵存放动态系统
    x[0,0] = x0

    for i in range(num_step):
        time[i+1,0] = i  # 记录信号时序
        x[i+1,0] = r*x[i,0]*(1-x[i,0])

    return time, x

# Mackey-Glass system (详情请参考：https://blog.csdn.net/qq_41137110/article/details/115772426)
# x0是系统的初始值，tau需要是一个正整数（positive integer）
def Mackey_Glass(x0=0.8,parameter=(10,0.2,0.1),tau=23,num_step=5000,step_length=1.0):
    alpha, beta, gamma = parameter     # 解压系统参数
    dt = step_length                   # 即每个时间步的步长

    time = np.zeros(num_step)  # 创建空矩阵以存放时序信号
    x = np.zeros((1,num_step))     # 创建一个数组记录系统的数据

    for i in range(tau):
        time[i] = i  # 记录时序信号
        x[0,i] = x0  # 将系统初始值赋予新的数据数组，需要准确地有tau步，锚定准确的初始条件，这样系统才能有初始动能启动

    # range(tau,tau+step) - 从tau到tau+step，不包含tau+step，即：tau, tau+1, tau+2, ..., tau+step-1
    for i in range(tau,num_step-tau):
        time[i] = i  # 记录时序信号
        dx_dt = -gamma*x[0,i-1]+beta*x[0,i-tau]/(1+x[0,i-tau]**alpha)
        x[0,i] = x[0,i-1]+dx_dt*dt    # x[t] = x[t-1]+(dx/dt)*dt

    return time, x

########################################################################################################################
# 2D dynamical systems

# 2D logistic hyperchaotic
def Logistic_hyperchaotic(origin=(0.5,0.5),a=0.98,num_step=5000):
    time = np.zeros(num_step)  # 创建空矩阵以存放时序信号
    trajectory = np.zeros((2, num_step), dtype=float)  # 创建一个数组记录系统的数据

    x, y = (np.zeros(num_step, dtype=float), np.zeros(num_step, dtype=float))  # 创建两个零数组方便后续计算
    x[0] = origin[0]  # 起点坐标赋值
    y[0] = origin[1]
    for i in range(1, num_step):
        time[i] = i  # 记录时序信号
        x[i] = np.sin(np.pi*(4*a*x[i-1]*(1-x[i-1]))+(1-a)*np.sin(np.pi*y[i-1]))  # 迭代更新坐标
        y[i] = np.sin(np.pi*(4*a*y[i-1]*(1-y[i-1]))+(1-a)*np.sin(np.pi*x[i]**2))

    trajectory[0,:] = x
    trajectory[1,:] = y

    return time, trajectory

########################################################################################################################
# 3D dynamical systems
# origin = [x,y,z] specific the starting point of the system, eg. origin = [0.1,0.1,0.1]
# parameter = [sigma, gamma, beta] ([Prandtl number, Rayleigh number, some number]) specific the parameters of the system

def RosslerAttractor(origin=(0.1,0,0),parameter=(0.1,0.1,14),num_step=5000,step_length=0.01):
    x, y, z = origin  # 解压参数
    a, b, c = parameter
    input = [x, y, z, a, b, c, step_length]
    x, y, z, a, b, c, dt = [float(n) for n in input]  # 将所有输入变量转换成浮点数，以防出错

    time = np.zeros(num_step, dtype=int)  # 创建空矩阵以存放时序信号
    trajectory = np.zeros((3, num_step), dtype=float)  # 创建一个（step+1）行，3列的零矩阵，每一行都是当前step的轨迹坐标，从origin开始
    trajectory[:, 0] = np.array(origin)  # 因为走step步，所以会有（step+1）个点

    for i in range(1, num_step):
        time[i] = i  # 记录信号时序

        x, y, z = trajectory[:, i - 1]  # 更新V1,V2,I的值

        dr_dt = np.array([-y-z, x+a*y, b+z*(x-c)])  # 计算轨迹变化导数：dr/dt = (dx/dt, dy/dt, dz/dt)
        # 更新x,y,z的值，得到下一个点的坐标, r[n+1] = r[n]+(dr/dt)*dt
        trajectory[:, i] = trajectory[:, i - 1] + dr_dt * dt

    return time, trajectory

# Lorenz-63 attractor, when gamma < 1, the attractor is the origin; when 1 <= gamma < 13.927, there two stable points
def Lorenz_63(origin=(3.05, 1.58, 15.62),parameter=(10.0, 29, 2.667),num_step=5000,step_length=0.01):
    x, y, z = origin                                       # 解压参数
    sigma, gamma, beta = parameter
    input = [x,y,z,sigma,gamma,beta,step_length]
    x,y,z,sigma,gamma,beta,dt = [float(n) for n in input]  # 将所有输入变量转换成浮点数，以防出错

    time = np.zeros(num_step, dtype=int)    # 创建空矩阵以存放时序信号
    trajectory = np.zeros((3,num_step),dtype=float)  # 创建一个（step+1）行，3列的零矩阵，每一行都是当前step的轨迹坐标，从origin开始
    trajectory[:,0] = np.array(origin)       # 因为走step步，所以会有（step+1）个点

    for i in range(1,num_step):
        time[i] = i  # 记录信号时序

        x, y, z = trajectory[:,i-1]  # 更新V1,V2,I的值

        dr_dt = np.array([sigma*(y-x), x*(gamma-z)-y, x*y-beta*z])  # 计算轨迹变化导数：dr/dt = (dx/dt, dy/dt, dz/dt)
        # 更新x,y,z的值，得到下一个点的坐标, r[n+1] = r[n]+(dr/dt)*dt
        trajectory[:, i] = trajectory[:, i - 1] + dr_dt * dt

    return time, trajectory

# 蔡氏电路 （Chua's circuit）
# origin = [V1,V2,I] specific the starting point of the system
# In this case, V1, V2 indicate the voltage applied to the capacitor C1 and C2 in the Chua's circuit, and I is the current flow through the inductor
# parameter = [alpha, beta, c, d], where alpha, beta is decided by the circuit components, alpha = C2/C1, beta = C2*R**2/L
# c and d is the parameter of the nonlinear resistor, we could assume c = Gb*R, d = Ga*R, where Ga and Gb is the slope of different section of the nonlinear resistor
def ChuaCircuit(origin=(0.1,0.1,0.1),parameter=(10,12.33,-0.544,-1.088),num_step=100000,step_length=0.001):
    V1, V2, I = origin                # 解压参数
    alpha, beta, c, d = parameter
    input = [V1, V2, I, alpha, beta, c, d, step_length]
    V1, V2, I, alpha, beta, c, d, dt = [float(n) for n in input]  # 将所有输入变量转换成浮点数，以防出错

    time = np.zeros(num_step, dtype=int) # 创建空矩阵以存放时序信号
    trajectory = np.zeros((3,num_step), dtype=float)  # 创建一个(3×num_step)的零矩阵，每一行都是当前step的轨迹坐标，从origin开始
    trajectory[:,0] = np.array(origin)  # 因为走step步，所以会有（step+1）个点

    for i in range(1,num_step):
        time[i] = i  # 记录信号时序

        V1, V2, I = trajectory[:,i-1]  # 更新V1,V2,I的值

        # 函數f描述了非線性電阻（即蔡氏二极管）的電子響應，並且它的形狀是依賴於它的元件的特定阻態
        f = c * V1 + 0.5 * (d - c) * (np.abs(V1 + 1) - np.abs(V1 - 1))

        dr_dt = np.array([alpha*(V2-V1-f), V1-V2+I, -beta*V2])   # 计算轨迹变化导数：dr/dt = (dV1/dt, dV2/dt, dI/dt)

        trajectory[:,i] = trajectory[:,i-1]+dr_dt*dt
        # 更新V1,V2,I的值，得到下一个点的坐标, r[n+1] = r[n]+(dr/dt)*dt

    return time, trajectory

##################################################### 辅助函数库 #########################################################
def Rearrange(data: np.ndarray) -> np.ndarray:
    '''
    这个函数可以重整以上函数的输出，方便数据分析
    :param data: dynamical system trajectory
    :return: rearranged dynamical system trajectory
    '''
    num_time_step = len(data)    # 数据的长度即为时间步的个数
    num_variable = len(data[0])  # 自变量的个数
    data_rearranged = np.zeros([num_variable+1,num_time_step])  # 创建一个行为（自变量数+1），列为时间步数的零矩阵
    data_rearranged[0] = np.linspace(0,num_time_step-1,num_time_step)  # 重整后的数据的第一行即为时间步
    for i in range(num_variable):
        data_rearranged[i+1] = data[:,i]  # 从第二列开始，每一列都是某一个自变量在不同时间步下的值

    return data_rearranged

def Add_noise(data: np.ndarray, noise_type: str='normal', SNR: float=20., **kwargs) -> np.ndarray:
    '''
    这个函数可以给动态系统轨迹数据添加噪声
    :param data: dynamical system trajectory
    :return: dynamical system trajectory with noise
    '''
    num_channel, len_traj = data.shape  # 获取动态系统轨迹数据的形状

    # 生成一个与数据大小相同的随机噪声基序列
    noise_dist_param = kwargs['noise_dist_param'] if 'noise_dist_param' in kwargs else (0, 0.1)  # 噪声分布参数
    if noise_type == 'normal':
        drift, width = noise_dist_param  # 获取噪声的均值和标准差
        noise_basic = np.random.normal(loc=drift, scale=width, size=(num_channel, len_traj))
    elif noise_type == 'uniform':
        inf, sup = noise_dist_param  # 获取下确界和上确界
        noise_basic = np.random.uniform(low=inf, high=sup, size=(num_channel, len_traj))
    else:
        raise ValueError('The noise type is not supported!')

    for i in range(num_channel):
        A_signal_avg = np.abs(data[i]).mean()        # 计算信号幅值的平均值
        amp_coeff = A_signal_avg/10**(SNR/20.)       # 计算噪声的幅值调节系数
        noise_basic[i] = noise_basic[i] * amp_coeff  # 根据指定信噪比调整噪声的幅值

    data_w_noise = data + noise_basic  # 将噪声添加到数据中

    return data_w_noise

if __name__ == '__main__':
    # Testing
    num_point = 3000

    tau = 30
    x0 = 0.8
    m_g = Mackey_Glass(x0,[10,0.2,0.1],tau,step=num_point,step_length=1)

    #x = [m_g[i+tau] for i in range(num_point-1)]
    #y = [m_g[i] for i in range(num_point-1)]
    #plt.plot(x,y)

    t = [i for i in range(num_point+tau)]
    plt.plot(t,m_g)