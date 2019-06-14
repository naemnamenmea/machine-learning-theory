import numpy as np
from numpy.linalg import norm


def find_min(funObj, w, maxEvals, verbose, *args):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w, *args)
    funEvals = 1.0

    alpha = 1.0
    while True:
        # Line-search using quadratic interpolation to find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1.0
            if f_new <= f - gamma * alpha * gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha ** 2) * gg / (2. * (f_new - f + alpha * gg))

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha * np.dot(y.T, g) / np.dot(y.T, y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        if verbose > 1:
            print("alpha: %.3f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w


class PCA:
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using SVD
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = np.mean(X, 0)
        X = X - self.mu

        U, s, V = np.linalg.svd(X)
        self.W = V[:self.k, :]
        return self

    def compress(self, X):
        X = X - self.mu
        Z = np.dot(X, self.W.transpose())
        return Z

    def expand(self, Z):
        X = np.dot(Z, self.W) + self.mu
        return X


class AlternativePCA:
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using gradient descent
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        n, d = X.shape
        k = self.k
        self.mu = np.mean(X, 0)
        X = X - self.mu

        # Randomly initial Z, W
        z = np.random.randn(n * k)
        w = np.random.randn(k * d)

        f = np.sum((np.dot(z.reshape(n, k), w.reshape(k, d)) - X) ** 2) / 2
        for i in range(50):
            f_old = f
            z = find_min(self._fun_obj_z, z, 10, False, w, X, k)
            w = find_min(self._fun_obj_w, w, 10, False, z, X, k)
            f = np.sum((np.dot(z.reshape(n, k), w.reshape(k, d)) - X) ** 2) / 2
            print('Iteration {:2d}, loss = {}'.format(i, f))
            if f_old - f < 1e-4:
                break

        self.W = w.reshape(k, d)
        return self

    def compress(self, X):
        n, d = X.shape
        k = self.k
        X = X - self.mu
        # We didn't enforce that W was orthogonal so we need to optimize to find Z
        z = np.zeros(n * k)
        z = find_min(self._fun_obj_z, z, 100, False, self.W.flatten(), X, k)
        Z = z.reshape(n, k)
        return Z

    def expand(self, Z):
        X = np.dot(Z, self.W) + self.mu
        return X

    def _fun_obj_z(self, z, w, X, k):
        n, d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z, W) - X
        f = np.sum(R ** 2) / 2
        g = np.dot(R, W.transpose())
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n, d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z, W) - X
        f = np.sum(R ** 2) / 2
        g = np.dot(Z.transpose(), R)
        return f, g.flatten()


class RobustPCA(AlternativePCA):

    def _fun_obj_z(self, z, w, X, k):
        # |Wj^TZi-Xij| = sqrt((Wj^TZi-Xij)^2 + 0.0001)
        # |ZW-X| = ((ZW-X)**2 + 0.0001)**0.5
        # f'z = 0.5((ZW-X)**2 + 0.0001)**-0.5 * 2(ZW-X)W**T
        n, d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z, W) - X
        f = np.sqrt(np.sum(R ** 2) + 0.0001)
        g = np.dot(R, W.transpose()) / (np.sum(R ** 2) + 0.0001) ** 0.5
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n, d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z, W) - X
        f = np.sqrt(np.sum(R ** 2) + 0.0001)
        g = np.dot(Z.transpose(), R) / (np.sum(R ** 2) + 0.0001) ** 0.5
        return f, g.flatten()


if __name__ == '__main__':
    import pylab as plt

    from prepare_data import FormData

    X, _, _, _ = FormData.prep_data(data="sinus")
    X = X.values
    n, d = X.shape
    print(X.shape)
    h, w = 64, 64  # height and width of each image

    # the two variables below are parameters for the foreground/background extraction method
    # you should just leave these two as default.

    k = 5  # number of PCs
    threshold = 0.04  # a threshold for separating foreground from background

    model = RobustPCA(k=k)
    model.fit(X)
    Z = model.compress(X)
    Xhat = model.expand(Z)

    # save 10 frames for illustration purposes
    # for i in range(10):
    #     plt.subplot(1, 3, 1)
    #     plt.title('Original')
    #     plt.imshow(X[i].reshape(h, w).T, cmap='gray')
    #     plt.subplot(1, 3, 2)
    #     plt.title('Reconstructed')
    #     plt.imshow(Xhat[i].reshape(h, w).T, cmap='gray')
    #     plt.subplot(1, 3, 3)
    #     plt.title('Thresholded Difference')
    #     plt.imshow(1.0 * (abs(X[i] - Xhat[i]) < threshold).reshape(h, w).T, cmap='gray')
    #     plt.show()
    #     utils.savefig('q2_highway_{:03d}.jpg'.format(i))


    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    n_components = 2

    pca = PCA(n_components)
    pca.fit(X)
    X_pca = pca.compress(X)

    colors = ['navy', 'turquoise', 'darkorange']

    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color, lw=2, label=target_name)
        plt.title('PCA')


    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

    plt.show()