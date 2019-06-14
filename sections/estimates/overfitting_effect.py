# Выйти когда внешний критерий начнет расти
# Вернуть также оптимальный набор признаков
# {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0} {1: 0.9081249999999967, 2: 0.8274999999999993, 3: 0.327499999999999, 4: 0.5775000000000026, 5: 0.5775000000000026, 6: 0.327499999999999, 7: 0.18749999999999964, 8: 0.5624999999999989, 9: 0.18749999999999964, 10: 0.5775000000000026, 11: 0.7074999999999998, 12: 0.625, 13: 0.8274999999999993}

from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.datasets import load_boston
from sklearn.tree import ExtraTreeRegressor
from math import inf, factorial
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from itertools import combinations

X, y = load_boston(return_X_y=True)
X = X[:20, :]
y = y[:20]
alg = ExtraTreeRegressor()
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
n = X.shape[1]

int_scores = {}
ext_scores = {}

for i in range(1, n + 1):
    int_score_tmp1 = inf
    ext_score_tmp1 = inf
    for features in combinations(range(n), i):
        X_cuted = X[:, features]
        int_score_tmp2 = inf
        ext_score_tmp2 = inf
        for train_index, test_index in cv.split(X_cuted):
            X_train, X_test = X_cuted[train_index], X_cuted[test_index]
            y_train, y_test = y[train_index], y[test_index]

            alg.fit(X_train, y_train)
            y_pred = alg.predict(X_train)
            error = mean_squared_error(y_train, y_pred)
            int_score_tmp2 = min(int_score_tmp2, error)

            y_pred = alg.predict(X_test)
            error = mean_squared_error(y_test, y_pred)
            ext_score_tmp2 = min(ext_score_tmp2, error)
        int_score_tmp1 = min(int_score_tmp1, int_score_tmp2)
        ext_score_tmp1 = min(ext_score_tmp1, ext_score_tmp2)
    int_scores[i] = int_score_tmp1
    ext_scores[i] = ext_score_tmp1

print(int_scores, ext_scores)
