import numpy as np
from scipy.optimize import leastsq
from scipy import optimize as op
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class model:
    """ This class of function is designed for fitting characterization result. """
    def __init__(self):
        self.name = model

    ###############################################################################################################
    # 线性回归（Linear Regression）
    # 此函数可利用线性回归拟合测试结果
    def LinearRegression(self,x,y,reshaping='False'):
        if reshaping == 'True':   # 默认为不需要reshape
            x = x.reshape(-1, 1)  # sklearn的输入需要是二维数组
            y = y.reshape(-1, 1)  # 若输入为一维数组，则需reshape为二维数组才可以进行拟合，reshape很重要
        else:
            pass

        # 利用sklearn进行线性回归拟合
        fitting_model = LinearRegression()  # 利用线性回归模块进行多项式回归
        fitting_model.fit(x, y)

        slope = fitting_model.coef_[0][0]  # 提取斜率
        intercept = fitting_model.intercept_[0]  # 提取截距

        return slope,intercept

    ###############################################################################################################
    # 多项式回归（Polynomial Regression）
    # 此函数可利用多项式回归拟合测试结果
    def PolynomialRegression(self,x,y,degree=3,reshaping='False'):
        if reshaping == 'True':   # 默认为不需要reshape
            x = x.reshape(-1, 1)  # sklearn的输入需要是二维数组
            y = y.reshape(-1, 1)  # 若输入为一维数组，则需reshape为二维数组才可以进行拟合，reshape很重要
        else:
            pass

        # 利用sklearn进行多项式回归拟合
        poly = PolynomialFeatures(degree=degree)  # 加入多项式函数特征
        poly.fit(x)
        x_poly = poly.transform(x)
        fitting_model = LinearRegression()  # 利用线性回归模块进行多项式回归
        fitting_model.fit(x_poly, y)

        fitting_result = fitting_model.coef_[0]  # 最后的拟合结果是二维数组，为方便后续运算及处理，我们将其转换为一维数组

        return fitting_result

    # 此函数可以对输入的多项式系数进行展开，还原多项式回归的结果
    # 自变量x可以是浮点数或者是一维数组，多项式回归的系数列表需要是一维数组
    def Y_PolyFit(self, x, coefficient):
        degree = len(coefficient)  # 多项式回归的阶数
        y_list = []
        for n in range(degree):
            y_list.append(coefficient[n]*x**n)  # 根据阶数，计算每一阶对函数总值的贡献
        y_mat = np.array(y_list)  # 将列表转换为二维数组，即矩阵
        y_total = y_mat.sum(axis=0)  # 进行每列的内部求和，即按列将整个矩阵求和成一行矩阵，结果为一维数组
        return y_total

    ###############################################################################################################
    # 拟合结果分析模块
    # 此函数可以对拟合结果进行分析，得到均方根误差跟决定因子
    def Evaluate(self,y,y_fit,reshaping='false'):
        if reshaping == 'True':          # 默认为不需要reshape
            y = y.reshape(-1, 1)         # 将实际数据转换为二维数组
            y_fit = y_fit.reshape(-1,1)  # 将拟合结果转换为二维数组
        else:
            pass

        MSE = mean_squared_error(y, y_fit)  # Mean Squared Error (MSE)
        R2 = r2_score(y, y_fit)             # Coefficient of determination (R^2)

        # 拟合结果评估
        print(r'Mean Squared Error (MSE): %.5f' % MSE)
        print(r'Coefficient of Determination (R^2): %.5f' % R2)

        return MSE, R2