import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prepare_data import FormData
from common import sse
from numpy.linalg import svd, inv
from sections.regression_analysis.linear_regression import LinearRegr


class RidgeRegr:
    def fit(self, X, y, k=None, alpha=0.4):

        V, s, Ut = svd(X, full_matrices=False)
        V = V[:, :k]
        s = np.diag(s)[:k, :k]
        Ut = Ut[:k, :]
        Vt, Xt, U = map(np.transpose, [V, X, Ut])
        tmp = inv(s.dot(s) - alpha * np.eye(s.shape[0]))
        self.alpha = U.dot(tmp).dot(s).dot(Vt).dot(y)
        self.alpha_norm = 0.
        for i in range(Vt.shape[0]):
            self.alpha_norm += Vt[i].dot(y) ** 2 / (s[i][i] ** 2 + alpha)
        self.Qconst = self.Q(X, y)

    def predict(self, X):
        return X.dot(self.alpha)

    def Q(self, X, y):
        tmp = X.dot(self.alpha) - y
        return np.transpose(tmp).dot(tmp)

    def depTauQ(self, X, y, k, l=0.1, r=0.9, step=0.1):
        ridgereg = RidgeRegr()
        res = []
        for tau in np.arange(l, r, step):
            ridgereg.fit(X=X, y=y, k=k, alpha=tau)
            res.append(list([tau, ridgereg.Q(X, y)]))
        return pd.DataFrame(res, columns=['tau', 'Q'])


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = FormData.prep_data(data="boston", train_size=0.75, test_size=1., shuffle=True,
                                                          use_control_sample=True)

    k = x_train.keys().size

    linreg_svd = LinearRegr()
    linreg_svd.fit(X=x_train, y=y_train, mode="svd", k=k)
    y_pred_linsvd = linreg_svd.predict(x_test)

    ridgereg = RidgeRegr()
    ridgereg.fit(X=x_train, y=y_train, k=k, alpha=20.4)
    y_pred_ridge = ridgereg.predict(x_test)

    depTau = ridgereg.depTauQ(X=x_train, y=y_train, k=k, l=0., r=2.5)
    plt.plot(depTau['tau'], depTau['Q'])
    plt.title('Dependence Q from tau')
    plt.xlabel('tau')
    plt.ylabel('Q(tau)')
    plt.savefig("images/dep_q_from_tau.png")
    # plt.show()

    table = pd.DataFrame(index=x_train.keys(), data={'before': linreg_svd.alfa, 'after': ridgereg.alpha})
    print(table, '\r\n')

    print("SSE (before Ridge): %.3f" % sse(y_pred_linsvd, y_test) +
          ";\t ||svd_alpha*||: %.3f" % linreg_svd.alfa_norm +
          ";\t svd_Q: %.3f" % linreg_svd.Q)

    print("SSE (after Ridge): %.3f" % sse(y_pred_ridge, y_test) +
          ";\t ||ridge_alpha*||: %.3f" % ridgereg.alpha_norm +
          ";\t ridge_Q: %.3f" % ridgereg.Qconst)
