# This code use scikit-learn to optimize the connection weight in reservoir computing algorithm models.
# https://scikit-learn.org/stable/

import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.linear_model import LassoCV,RidgeCV,Ridge,ElasticNetCV,orthogonal_mp,OrthogonalMatchingPursuit

# the method of training the readout matrix
# anova - analysis of variance (方差分析)
def WeightOptimizing(train_data, R_state, index=0, k=0.8):
    W_out = np.zeros((R_state.shape[1], train_data.shape[1]))
    if index == 0:
        W_out = np.dot(np.linalg.pinv(R_state), train_data)  # Dr*N
    else:
        alphas = 10 ** np.linspace(-4, 2, 7)
        if index == 1:
            base_cv = LassoCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        if index == 2:
            base_cv = RidgeCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)
        if index == 3:
            base_cv = ElasticNetCV(alphas=alphas, fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)

        if index == 4:
            anova_filter = SelectKBest(f_regression,
                                       k=int(R_state.shape[1] * k))  # int(self.n_reservoir*0.8 ))#k_number)

        if index == 5:
            base = linear_model.LinearRegression(fit_intercept=True)
            anova_filter = RFECV(base)
        if index == 6:
            base_cv = OrthogonalMatchingPursuit(fit_intercept=False)
            anova_filter = SelectFromModel(base_cv)

        clf = Pipeline([
            ('feature_selection', anova_filter),
            ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
        ])
        for X_i in range(train_data.shape[1]):
            clf.fit(R_state, train_data[:, X_i])
            W_out[clf.named_steps['feature_selection'].get_support(), X_i] = clf.named_steps['Linearregression'].coef_
    return W_out