import numpy as np
from numpy import dot, transpose
from numpy.linalg import inv, svd, pinv
from prepare_data import FormData
from common import sse
import common as com


class LinearRegr:

    def fit(self, X, y, mode="casual", k=None):

        self.mode = mode
        self.k = k or X.shape[1]
        if mode == "svd":
            V, s, Ut = svd(X, full_matrices=False)
            V = V[:, :self.k]
            s = np.diag(s)[:self.k, :self.k]
            si = inv(s)
            Ut = Ut[:self.k]
            Vt, U = map(np.transpose, [V, Ut])
            Xpinv = U.dot(si.dot(Vt))
            tmp = si.dot(Vt).dot(y)
            self.alfa_norm = transpose(tmp).dot(tmp)
        else:
            Xpinv = pinv(X)

        self.alfa = dot(Xpinv, y)
        self.Q = sse(X.dot(Xpinv).dot(y), y)

        if mode == "casual":
            self.alfa_norm = sum(self.alfa ** 2)

    def predict(self, X):
        return X.dot(self.alfa)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin) * np.random.rand(n) + vmin


def fun2(f1, f2, a1, a2):
    return a1 * f1 + a2 * f2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x_train, y_train, x_test, y_test = FormData.prep_data(data="boston", train_size=.25, shuffle=True,
                                                          use_control_sample=True, test_size=0.1)

    features = [11, 12]  # 0,1,2 / # 4,5,7,12
    k = x_train.keys().size

    X_train_cuted = x_train.iloc[:, features]
    X_test_cuted = x_test.iloc[:, features]

    lin_regr_casual = LinearRegr()
    lin_regr_svd = LinearRegr()

    lin_regr_casual.fit(X=x_train, y=y_train, mode="casual")
    lin_regr_svd.fit(X=X_train_cuted, y=y_train, mode="svd", k=k)

    alpha_dict_casual = dict(zip(x_train.keys(), lin_regr_casual.alfa))
    print('All %d features: ' % (x_train.keys().size), alpha_dict_casual, '\r\n')

    y_pred_casual_full = lin_regr_casual.predict(X=x_test)

    alpha1 = lin_regr_svd.alfa[0]
    alpha2 = lin_regr_svd.alfa[1]

    alpha_dict_svd = dict(zip(X_train_cuted.keys(), lin_regr_svd.alfa))
    print('Selected %d features: ' % (features.__len__()), alpha_dict_svd, '\r\n')

    y_pred_svd_cuted = lin_regr_svd.predict(X=X_test_cuted)

    print('SSE OLS with all features: %.3f' % sse(y_test, y_pred_casual_full))
    print('SSE OLS+SVD with 2 features: %.3f' % sse(y_test, y_pred_svd_cuted))

    fig = plt.figure(figsize=(com.W_inch, com.H_inch))
    ax = fig.add_subplot(111, projection='3d')
    f1 = np.array(X_test_cuted.iloc[:, 0])
    f2 = np.array(X_test_cuted.iloc[:, 1])
    X, Y = np.meshgrid(f1, f2)
    zs = np.array([fun2(f1, f2, alpha1, alpha2) for f1, f2 in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    ax.set_xlabel(x_train.keys()[features[0]])
    ax.set_ylabel(x_train.keys()[features[1]])
    ax.set_zlabel('y')
    ax.scatter(f1, f2, y_test)

    # plt.savefig("images/linear_regression_svd_k=" + str(k) + ".png")
    plt.show()
