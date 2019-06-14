import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy import array, dot, transpose
from numpy.linalg import inv
from sklearn import datasets


def linear_regression(x_train, y_train, x_test):
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)

    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)

    predictions = []
    x_test = np.array(x_test)
    for i in x_test:
        components = w[1:] * i
        predictions.append(sum(components) + w[0])
    predictions = np.asarray(predictions)
    return predictions



dataset = datasets.load_boston()
target = dataset.target

f_plot = pd.DataFrame(dataset.data, columns=dataset.feature_names)
f_plot['Target'] = target

train_size = int(round(506 * 0.75))  # Training set size: 75% of full data set.

A = f_plot.values

temp = A.T.dot(A)
S, V = np.linalg.eig(temp)
S = np.diag(np.sqrt(S))

U = A.dot(V).dot(np.linalg.inv(S))
reconstructed_2 = U.dot(S).dot(V.T)
df_2 = pd.DataFrame(reconstructed_2, columns=f_plot.columns)

train_2 = df_2[:train_size]
test_2 = df_2[train_size:]

x_train_2 = train_2.drop('Target', axis=1)
y_train_2 = train_2['Target']

x_test_2 = test_2.drop('Target', axis=1)
y_test_2 = test_2['Target']

res_2 = linear_regression(x_train_2, y_train_2, x_test_2)

print(sum((y_test_2 - res_2)**2))