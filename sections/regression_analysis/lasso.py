from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy
import numpy as np


class Lasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        beta = np.zeros(X.shape[1])
        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        for iteration in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = deepcopy(beta)
                tmp_beta[j] = 0.0
                r_j = y - np.dot(X, tmp_beta)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.alpha * X.shape[0]

                beta[j] = self._soft_thresholding_operator(arg1, arg2) / (X[:, j] ** 2).sum()

                if self.fit_intercept:
                    beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self

    def predict(self, X):
        y = np.dot(X, self.coef_)
        if self.fit_intercept:
            y += self.intercept_ * np.ones(len(y))
        return y


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_validate
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score
    from prepare_data import FormData
    from sklearn.model_selection import cross_val_score

    X, y, _, _ = FormData.prep_data(data="diabetes")
    X = X.values
    y = y.values

    res = []
    for alpha in np.arange(-1., 3., 0.3):
        model = Lasso(alpha=alpha, max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)

        # print('Coefficients: \n', model.coef_)
        res.append(list([alpha, mean_squared_error(y, y_pred)]))

    res2 = pd.DataFrame(res, columns=['alpha', 'MSE'])

    plt.plot(res2['alpha'].values, res2['MSE'].values, label='LASSO', color='red')
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    plt.show()
    plt.savefig('images/lasso.png')
