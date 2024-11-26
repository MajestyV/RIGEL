# This code is designed for optimizing the output weight matrix of Reservoir Computing (RC) model.

import numpy as np
import sklearn.linear_model as sk_fitting
from sklearn.metrics import mean_squared_error, r2_score

# Ridge regression（岭回归），又被称为吉洪诺夫正则化（Tikhonov regularization）
def RIDGE(input, output, alpha=1.0, show_evaluation=True):
    input = np.array([np.array(input[n]) for n in range(len(input))])  # 确保输入是一个二维数组
    output = np.array([np.array(output[n]) for n in range(len(output))])  # 确保输出是一个二维数组

    ridge = sk_fitting.Ridge(alpha=alpha)  # 输入正则化系数
    ridge.fit(input, output)

    output_predict = ridge.predict(input)

    # 拟合结果评估
    if show_evaluation:
        print('Mean Squared Error (MSE): %.2f' % mean_squared_error(output, output_predict))
        print('Coefficient of determination (R^2): %.2f' % r2_score(output, output_predict))
    else:
        pass

    return ridge.coef_, ridge.intercept_

# LASSO（least absolute shrinkage and selection operator）回归
def LASSO(input, output, alpha=0.025, show_evaluation=True):
    input = np.array([np.array(input[n]) for n in range(len(input))])  # 确保输入是一个二维数组
    output = np.array([np.array(output[n]) for n in range(len(output))])  # 确保输出是一个二维数组

    lasso = sk_fitting.Lasso(alpha=alpha)  # 输入正则化系数
    lasso.fit(input,output)

    output_predict = lasso.predict(input)

    # 拟合结果评估
    if show_evaluation:
        print('Mean Squared Error (MSE): %.2f' % mean_squared_error(output, output_predict))
        print('Coefficient of determination (R^2): %.2f' % r2_score(output, output_predict))
    else:
        pass

    return lasso.coef_, lasso.intercept_

