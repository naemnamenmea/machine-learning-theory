from math import ceil
import numpy as np
from scipy import linalg


def lowess(x, y, f=2. / 3., iter=3):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y * x), np.sum(weights * y)]) # почему x_i, а не d_i = x_i - x?
            A = np.array([[np.sum(weights * x * x), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] * x[i] + beta[1] # beta[0] * (u - x[i]) + beta[1] ??

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1) # если выходим за границы (-1;1), то должно быть 0
        delta = (1 - delta ** 2) ** 2

    return yest