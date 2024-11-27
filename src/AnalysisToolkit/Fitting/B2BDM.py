import numpy as np
from scipy.optimize import leastsq
from scipy import optimize as op
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class model:
    """ Fitting the I-V curves of two terminal devices using back-to-back diode model (B2BDM). """
    def __init__(self,**kwargs):
        self.name = model

        # 全局参数（Global parameters）
        self.T = kwargs['T'] if 'T' in kwargs else 300.0  # 温度（默认为300）, [=] K
        self.kB = 1.380649e-23                            # 玻尔兹曼常数, [=] J/K
        self.q = 1.602176634e-19                          # 基本电荷, [=] C

    # This function is the standard I=f(V) curve of an ideal diode
    def Current_diode(self,V,parameter):
        I0, n = parameter  # 从输入变量parameter中将各个参数解压出来
        q, kB, T = [self.q, self.T, self.kB]  # 引入全局变量，并解压赋值给本地变量
        alpha = q / (n * kB * T)  # 把一些常用的系数预先算好
        I = I0*(np.exp(alpha*V)-1)
        return I

    # This function is the standard I=f(V) curve of symmetric back-to-back diode devices.
    def Current_sym(self,V,parameter):
        Io1, Io2, n = parameter  # 从输入变量parameter中将各个参数解压出来
        q, kB, T = [self.q, self.T, self.kB]  # 引入全局变量，并解压赋值给本地变量
        alpha = q/(2.0*n*kB*T)  # 把一些常用的系数预先算好
        I = 2.0*Io1*Io2*np.sinh(alpha*V)/(Io1*np.exp(-alpha*V)+Io2*np.exp(alpha*V))
        return I

    # This function is designed for fitting real asymmetric back-to-back diode devices.
    def Voltage_asym(self,I,parameter):
        Io1, Io2, n1, n2, Rs = parameter  # 从输入变量parameter中将各个参数解压出来
        q, kB, T = [self.q, self.T, self.kB]  # 引入全局变量，并解压赋值给本地变量
        # In python, np.log(x) is the Natural Log (base e log, ln) of x
        V1 = n1*kB*T*np.log(1.0+I/Io1)/q  # Diode_1分得的电压
        V2 = -n2*kB*T*np.log(1.0-I/Io2)/q  # Diode_2分得的电压
        Vs = I*Rs  # 寄生电阻分得的电压
        return V1+V2+Vs

    ################################################################################################################
    # 数据拟合模块
    # 这个函数采用了scipy.optimize中的curve_fit函数进行拟合，算法是非线性最小二乘法
    def Fitting(self,target_function,x_data,y_data):
        return op.curve_fit(target_function,x_data,y_data)

    # 这个函数采用了scipy中的最小二乘法拟合，理论上跟上面的函数是一模一样的
    def Least_square(self,initial_parameter,x_data,y_data):
        def error(parameter,x,y):  # 定义误差函数
            return self.Voltage_asym(x,parameter)-y

        optimized_parameter = leastsq(error,initial_parameter,args=(x_data,y_data))[0]  # 利用最小二乘法优化参数

        return optimized_parameter

    def Least_square_diode(self, initial_parameter, x_data, y_data):
        def error(parameter, x, y):  # 定义误差函数
            return self.Current_diode(x, parameter) - y

        optimized_parameter = leastsq(error, initial_parameter, args=(x_data, y_data))[0]  # 利用最小二乘法优化参数

        return optimized_parameter

    def Least_square_B2BDM(self, initial_parameter, x_data, y_data):
        def error(parameter, x, y):  # 定义误差函数
            return self.Current_sym(x, parameter) - y

        optimized_parameter = leastsq(error, initial_parameter, args=(x_data, y_data))[0]  # 利用最小二乘法优化参数

        return optimized_parameter

    # 通过Taylor展开拟合器件性能
    # sklearn的输入需要是二维数组
    def DeviceFitting_polynomial(self, x_data, y_data, degree=3, reshaping='false'):
        if reshaping == 'True':  # 默认为不需要reshape
            x_data = x_data.reshape(-1, 1)  # 对输入一维数组reshape为二维数组才可以进行拟合
            y_data = y_data.reshape(-1, 1)  # reshape很重要
        else:
            pass

        poly = PolynomialFeatures(degree=degree)  # 加入多项式函数特征
        poly.fit(x_data)
        x_data_poly = poly.transform(x_data)
        parameter = LinearRegression()
        parameter.fit(x_data_poly, y_data)

        fitting_result =parameter.coef_[0]  # 最后的拟合结果是二维数组，为方便后续运算及处理，我们将其转换为一维数组

        return fitting_result

    # 通过Taylor展开拟合的器件性能,自变量V需要是浮点数或者是一维数组
    # 泰勒展开的系数列表需要是一维数组
    def I_Taylor(self, V, coefficient):
        degree = len(coefficient)  # 泰勒展开的阶数
        I_list = []
        for n in range(degree):
            I_list.append(coefficient[n] * V ** n)  # 根据阶数，计算每一阶对函数总值的贡献
        I_mat = np.array(I_list)  # 将列表转换为二维数组，即矩阵
        I_total = I_mat.sum(axis=0)  # 进行每列的内部求和，即按列将整个矩阵求和成一行矩阵，结果为一维数组
        return I_total

if __name__ == '__main__':
    pass