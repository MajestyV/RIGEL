# 以下是通过多项式回归（polynomial regression）拟合的器件激活函数

import numpy as np

# 多项式回归拟合的器件性能（Device performance fitted by polynomial regression），系数列表应从零阶开始，由低到高排列
# 自变量x可以是浮点数或者是一维数组，多项式回归的系数列表需要是一维数组
def Polynomial(x,coefficient):
    order = len(coefficient)  # 多项式回归的阶数
    y_list = []
    for n in range(order):
        y_list.append(coefficient[n]*np.power(x,n))  # 根据阶数，计算每一阶对函数总值的贡献
    y_matrix = np.array(y_list)  # 将列表转换为二维数组
    y = y_matrix.sum(axis=0)     # 进行每列的内部求和，即按列将整个矩阵求和成一行矩阵，结果为一维数组
    return y

# 通过Taylor展开拟合的器件性能,自变量V需要是浮点数或者是一维数组
# The default coefficient is obtained in oue of author's previous work
def I_Taylor(V, coefficient=(0, -0.0606270243, 0.00364103237, 0.140685043, 0.00988703156, -0.00824646444,
                             -0.000618645284, 0.000257831028, 0.000011526794, -0.00000315380367)):
    degree = len(coefficient)  # 泰勒展开的阶数
    I_list = []
    for n in range(degree):
        I_list.append(coefficient[n] * V ** n)  # 根据阶数，计算每一阶对函数总值的贡献
    I_mat = np.array(I_list)  # 将列表转换为二维数组，即矩阵
    I_total = I_mat.sum(axis=0)  # 进行每列的内部求和，即按列将整个矩阵求和成一行矩阵，结果为一维数组
    return I_total

def I_Taylor_w_deviation(V: float or np.ndarray, coefficient: tuple = (0, -0.0606270243, 0.00364103237, 0.140685043,
                                                                       0.00988703156, -0.00824646444, -0.000618645284,
                                                                       0.000257831028, 0.000011526794, -0.00000315380367),
                         deviation: tuple = (-1.49e-3,14.32e-3)):

    deviation_coeff = np.random.normal(loc=deviation[0], scale=deviation[1], size=1)

    degree = len(coefficient)  # 泰勒展开的阶数
    I_list = []
    for n in range(degree):
        I_list.append(coefficient[n] * V ** n)  # 根据阶数，计算每一阶对函数总值的贡献
    I_mat = np.array(I_list)  # 将列表转换为二维数组，即矩阵
    I_total = I_mat.sum(axis=0)  # 进行每列的内部求和，即按列将整个矩阵求和成一行矩阵，结果为一维数组
    return I_total*(1.0+deviation_coeff)


# 以下是通过指数型背靠背二极管模型（B2BDM）特性拟合的器件激活函数（效果一般）
# 此函数是为数组型输入而设的非线性器件激活函数
def Nonlinear_IV(V, parameters):
    # 常规物理参数（Global parameters）
    T = 300.0  # 温度（默认为300）, [=] K
    kB = 1.380649e-23  # 玻尔兹曼常数, [=] J/K
    q = 1.602176634e-19  # 基本电荷, [=] C

    I1, I2, n, s_func = parameters  # 解压初始参数
    alpha = q/(2*n*kB*T)

    # 利用np.piecewise实现分段函数，这样可以大大缩短运算时间
    # np.piecewise的输入需要是零维或一维数组，因此要分开对二维数组中的每行进行历遍
    # 同时，要注意piecewise不会更改输入跟输出的数据格式，即输入是数组，输出是数组；输入是整型，输出是整型；若输入是整型数组，则输出也是整型数组！
    I = []
    for i in range(len(V)):
        I_row = np.piecewise(V[i], [V[i] < 0, V[i] >= 0],
                             [lambda x: -2*s_func*I1*I2*np.sinh(-alpha*x)/(I1*np.exp(alpha*x)+I2*np.exp(-alpha*x)),  # 负电压部分
                              lambda x: 2*s_func*I1*I2*np.sinh(alpha*x)/(I1*np.exp(-alpha*x)+I2*np.exp(alpha*x))])   # 正电压部分
        I.append(I_row)

    return np.array(I)

# 此函数是为单点输入（单浮点数输入）而设的非线性器件激活函数
def Nonlinear_IV_float(V, parameters):
    # 常规物理参数（Global parameters）
    T = 300.0  # 温度（默认为300）, [=] K
    kB = 1.380649e-23  # 玻尔兹曼常数, [=] J/K
    q = 1.602176634e-19  # 基本电荷, [=] C

    I1, I2, n, s_func = parameters  # 解压初始参数
    alpha = q/(2*n*kB*T)

    # 利用np.piecewise实现分段函数，这样可以大大缩短运算时间
    # np.piecewise的输入需要是零维或一维数组，因此要分开对二维数组中的每行进行历遍
    # 同时，要注意piecewise不会更改输入跟输出的数据格式，即输入是数组，输出是数组；输入是整型，输出是整型；若输入是整型数组，则输出也是整型数组！
    I = np.piecewise(V, [V < 0, V >= 0],
                     [lambda x: -2*s_func*I1*I2*np.sinh(-alpha*x)/(I1*np.exp(alpha*x)+I2*np.exp(-alpha*x)),  # 负电压部分
                      lambda x: 2*s_func*I1*I2*np.sinh(alpha*x)/(I1*np.exp(-alpha*x)+I2*np.exp(alpha*x))])   # 正电压部分

    return I

if __name__ == '__main__':
    V = np.array([[ -5.81547366,-12.03811492,7.67718305,-6.1912044, 9.20636327]])
    I = Nonlinear_IV(V,[2.036589219,5.65E-03,3.13E+01,1.0])

    print(I)