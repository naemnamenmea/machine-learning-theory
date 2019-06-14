import numpy as np
import common as cmn


class NadarayaWatson():
    def __init__(self, h=1, kerne=cmn.kernel.epanechnikov, metric=cmn.euclidian):
        self.__h = h
        self.__kerne = kerne
        self.__metric = metric

    def set_params(self, h=None):
        self.__h = h or self.__h

    def _loo(self, X, y,h):
        len = X.size
        res = 0
        saved_h = self.__h
        self.__h = h
        for i in range(len):
            newX = np.delete(X, i)
            newY = np.delete(y, i)
            pred = self.predict(X_test=X[i], X=newX, y=newY)
            se = 0
            if not np.isnan(pred):
                se = (pred - y[i]) ** 2
            res += se
        self.__h = saved_h
        return res

    def predict(self, X, y, X_test):
        Ya = np.zeros(X_test.size)
        for i in range(X_test.size):
            numerator = 0.
            denominator = 0.
            for j in range(X.size):
                core_value = self.__kerne(self.__metric(X_test[i], X[j]) / self.__h)
                denominator += core_value
                numerator += y[j] * core_value
            with np.errstate(divide='ignore', invalid='ignore'):
                Ya[i] = numerator / denominator
        return Ya

    def get_h(self):
        return self.__h

    def fit(self, X, y, min_h=0.1, max_h=2., step_h=0.15):
        LOO_min = self._loo(X=X,y=y,h=self.__h)
        h_opt = self.__h
        res_h = []
        res_loo = []
        for h in cmn.frange(min_h, max_h, step_h):
            LOO = self._loo(X=X,y=y,h=h)
            res_h.append(h)
            res_loo.append(LOO)
            if LOO < LOO_min:
                LOO_min = LOO
                h_opt = h
        self.__h = h_opt
        return {'h_opt': h_opt, 'h': res_h, 'LOO': res_loo}

if __name__ == '__main__':
    pass