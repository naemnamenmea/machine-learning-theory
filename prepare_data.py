import sklearn.datasets as skd
import pandas as pd
import numpy as np
import os
import sklearn.preprocessing as prep


class FormData:
    @staticmethod
    def diabetes():
        diabetes = skd.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        df = pd.DataFrame(diabetes_X)
        df['y'] = pd.Series(diabetes.target)
        return df

    @staticmethod
    def boston():
        boston = skd.load_boston()
        df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        df['MEDV'] = pd.Series(boston.target)
        return df

    @staticmethod
    def sinus():
        X = np.linspace(0, 6, 100)
        y = 1 + 2 * np.sin(X)
        yhat = y + .5 * np.random.normal(size=len(X))

        # Create feature matrix
        tX = np.array([X]).T
        tX = np.hstack((tX, np.power(tX, 2), np.power(tX, 3)))

        df = pd.DataFrame(tX)
        df['MEDV'] = pd.Series(yhat)
        return df

    @staticmethod
    def prep_data(data="boston", shuffle=False, path_to_csv="", use_control_sample=False, train_size=0.75,
                  test_size=0.2, scale=False):

        if path_to_csv.__eq__(""):
            df = getattr(registry["FormData"], data)()
        else:
            df = pd.read_csv(path_to_csv, index_col=0)
        n = df.shape[1] - 1
        if False:
            df = (df - df.mean()) / df.std()
        if shuffle == True:
            df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
        train_size = int(round(len(df) * train_size))
        test_size = int(round((len(df) - train_size) * test_size))
        train = df.iloc[:train_size]
        if use_control_sample == True:
            test = df.iloc[train_size:(train_size + test_size)].reset_index(drop=True)
        else:
            test = train

        x_train = train.drop(train.columns[n], axis=1)
        if scale == True:
            x_train = pd.DataFrame(prep.scale(x_train), columns=x_train.keys())
        y_train = train.iloc[:, n]

        x_test = test.drop(test.columns[n], axis=1)
        if scale == True:
            x_test = pd.DataFrame(prep.scale(x_test), columns=x_test.keys())
        y_test = test.iloc[:, n]

        return x_train, y_train, x_test, y_test


registry = {'FormData': FormData}
os.makedirs("images", exist_ok=True)
