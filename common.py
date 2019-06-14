import numpy as np
import math
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

W_inch = 17
H_inch = 10

def form_image(path, source, name, scope='global', ext=None):
    base = ''
    if scope == 'global':
        base += ROOT_PATH
    elif scope == 'local':
        base += os.path.dirname(path)
    else:
        pass

    return base + '/images/' + file_name(source) + '_' + name + ('.' + ext if ext else '')


def file_name(full_path):
    return os.path.splitext(os.path.basename(full_path))[0]


class kernel:
    @staticmethod
    def epanechnikov(u):
        if math.fabs(u) <= 1:
            return (3. / 4) * (1. - u * u)
        else:
            return 0.

    @staticmethod
    def gaussian(u):
        return math.exp(-1. / 2 * u * u) / math.sqrt(2 * math.pi)

    @staticmethod
    def uniform(u):
        if math.fabs(u) <= 1:
            return 0.5
        else:
            return 0.

    @staticmethod
    def quartic(u):
        if math.fabs(u) <= 1:
            return 15. / 16. * (1 - u ** 2) ** 2
        else:
            return 0.


def sse(X, X_pred):
    return sum((X - X_pred) ** 2)


def euclidian(a, b):
    return math.sqrt(np.sum(np.subtract(b, a) ** 2))


def genX(l_lim, r_lim, size):
    # return np.random.uniform(low = l_lim, high = r_lim, size=(size,))
    return np.linspace(l_lim, r_lim, size).astype(int)


def genY(X, noise=0., n_outliers=0, outlier_noise=35., random_state=0):
    Y = list(map(lambda x: math.sqrt(math.fabs(x)) * math.sin(x / 3.), X))
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(X.size)
    outliers = rnd.randint(0, X.size, n_outliers)
    error[outliers] *= outlier_noise
    return Y + error


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step