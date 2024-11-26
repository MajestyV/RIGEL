# 此代码可以进行权重优化，包括Tikhonov regularization和scikit-learn的机器学习算法

import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV,OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error, r2_score

# 基于scikit-learn的训练模块 ------------------------------------------------------------------------------------------

def Tikhonov_regularization(model_output: np.ndarray, target: np.ndarray, alpha: float=1.0, show_evaluation: bool=True):
    '''
    此函数可以计算训练集输入生成的所有储层态，并利用Tikhonov regularization计算输出权重，并计算网络输出
    详情请参考：https://en.wikipedia.org/wiki/Tikhonov_regularization
    '''

    ridge = linear_model.Ridge(alpha=alpha)  # 输入正则化系数
    ridge.fit(model_output, target)  # 利用Tikhonov正则化（即岭回归）计算输出权重

    W_out, threshold = (ridge.coef_, ridge.intercept_)  # 获取优化得到的权重和阈值

    model_predict = np.dot(model_output, W_out.T) + threshold  # 网络的输出，可用于计算各类统计指标，分析训练结果

    # 拟合结果评估
    if show_evaluation:
        print('Mean Squared Error (MSE): %.2f' % mean_squared_error(model_predict, target))
        print('Coefficient of determination (R^2): %.2f' % r2_score(model_predict, target))
    else:
        pass

    return (W_out.T, threshold), model_predict

def Train_sklearn(model_output: np.ndarray, target: np.ndarray, opt_algorithm: int=2, fit_intercept: bool=True,
                  alpha_set: np.ndarray=10**np.linspace(-4, 2, 7), show_evaluation: bool=True, **kwargs):
    '''
    此函数可以通过scikit-learn的机器学习算法，对网络进行训练，得到输出权重矩阵，并计算网络的输出
    '''

    if opt_algorithm == 0:
        base_cv = linear_model.LinearRegression(fit_intercept=fit_intercept)  # 最小二乘法线性回归
        anova_filter = SelectFromModel(base_cv)
    elif opt_algorithm == 1:
        base_cv = LassoCV(alphas=alpha_set, fit_intercept=fit_intercept)  # LASSO 回归
        anova_filter = SelectFromModel(base_cv)
    elif opt_algorithm == 2:
        base_cv = RidgeCV(alphas=alpha_set, fit_intercept=fit_intercept)  # 岭回归，又称Tikhonov regularization
        anova_filter = SelectFromModel(base_cv)
    elif opt_algorithm == 3:
        base_cv = ElasticNetCV(alphas=alpha_set, fit_intercept=fit_intercept)
        anova_filter = SelectFromModel(base_cv)
    elif opt_algorithm == 4:
        k = kwargs['k'] if 'k' in kwargs else 0.8
        anova_filter = SelectKBest(f_regression,k=int(model_output.shape[1] * k))  # int(self.n_reservoir*0.8 ))#k_number)
    elif opt_algorithm == 5:
        base = linear_model.LinearRegression(fit_intercept=fit_intercept)
        anova_filter = RFECV(base)
    elif opt_algorithm == 6:
        base_cv = OrthogonalMatchingPursuit(fit_intercept=fit_intercept)
        anova_filter = SelectFromModel(base_cv)
    else:
        print('Please select a valid optimization algorithm ! ! !')
        exit()

    # 利用scikit-learn的pipeline机制，将特征选择和线性回归模型连接在一起
    model = Pipeline([('feature_selection', anova_filter),
                      ('Linearregression', linear_model.LinearRegression(fit_intercept=fit_intercept))])

    output_dim, target_dim = (model_output.shape[1], target.shape[1])  # 获取各个维度
    W_out = np.zeros((output_dim, target_dim))  # 创建一个零数组用于存放输出权重
    threshold = np.zeros(target_dim, dtype=float)  # 创建一个零向量用于存放阈值向量

    for i in range(target_dim):
        model.fit(model_output, target[:,i])
        W_out[model.named_steps['feature_selection'].get_support(),i] = model.named_steps['Linearregression'].coef_
        threshold[i] = model.named_steps['Linearregression'].intercept_

    model_predict = np.dot(model_output, W_out) + threshold  # 网络的输出，可用于计算各类统计指标，分析训练结果

    # 拟合结果评估
    if show_evaluation:
        print('Mean Squared Error (MSE): %.2f' % mean_squared_error(model_predict, target))
        print('Coefficient of determination (R^2): %.2f' % r2_score(model_predict, target))
    else:
        pass

    return (W_out, threshold), model_predict  # 转置以适配ASHEN的数据结构