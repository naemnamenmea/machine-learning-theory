from sklearn.utils import resample, shuffle

def fit_in_range(x, lowest, highest):
    if x < lowest:
        return lowest
    elif x > highest:
        return highest
    else:
        return x


class Bootstrap():
    def __init__(self, n_splits=20, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.train_size = fit_in_range(train_size or 0.75, 0.5, 0.9)
        self.test_size = 1 - self.train_size
        self.random_state = random_state

    def split(self, X, *kargs, **kwargs):
        length = len(X)
        ix_total = [i for i in range(length)]
        n_samples = int(length * self.train_size)
        for i in range(self.n_splits):
            ix_total = shuffle(ix_total)
            ix_train = ix_total[:n_samples]
            ind_test = ix_total[n_samples:]

            ind_train = resample(ix_train, replace=True, n_samples=length,
                                 random_state=self.random_state)
            yield ind_train, ind_test


if __name__ == '__main__':
    from sklearn.model_selection import LeaveOneOut, LeavePOut, ShuffleSplit, RepeatedKFold
    from sklearn.model_selection import cross_validate
    import pandas as pd
    import sklearn.datasets as datasets
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    X, y = datasets.load_digits(return_X_y=True)
    X, y = shuffle(X, y)
    n_samples = min(228, X.shape[0])
    X = X[:n_samples, :]
    y = y[:n_samples]

    # alg = SVR(gamma='scale')
    alg = SVC()
    # alg = KNeighborsClassifier()
    # alg = DecisionTreeClassifier()

    cv_tq_fold_cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=42)
    cv_shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2)
    cv_bootstrap = Bootstrap(X.shape[0])
    cv_loo = LeaveOneOut()
    cv_ccv = LeavePOut(p=1) # n_samples - 1
    cv_group = [cv_loo, cv_shuffle_split, cv_ccv, cv_bootstrap, cv_tq_fold_cv]

    res = []
    for cv in cv_group:
        scores = cross_validate(alg, X, y, cv=cv, return_train_score=True)
        score = abs(scores['test_score'].mean())
        res.append([type(cv).__name__, score])
    table = pd.DataFrame(data=res, columns=['cv', 'score'])

    print('alg: ', type(alg).__name__)
    print('X_size: ', len(X), '; features: ', X.shape[1])
    print(table)
