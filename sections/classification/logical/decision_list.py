# др. название SCM

# стратегии выбора следующего класса
# Выбор предиката Ф
# критерий информативности
# метод поиска информативных предикатов

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from id3 import Id3Estimator
    from id3 import export_graphviz

    bunch = load_breast_cancer()
    estimator = Id3Estimator()
    estimator.fit(bunch.data, bunch.target)
    # export_graphviz(estimator,'')
    from common import form_image
    import os
    print(form_image(os.path.curdir(),__file__,'name'))


    # import numpy as np
    # from functional import seq
# ## Объявление выборки
#     X = [
#         [1, "a", False],
#         [3, "g", True],
#         [9, "c", False],
#         [3, "f", True],
#         [3, "a", True],
#         [5, "j", True],
#         [3, "a", False],
#         [8, "j", True],
#         [4, "j", True],
#         [5, "a", False]
#     ]
#     y = [1, 1, 1, 0, 1, 1, 0, 0, 0, 1]
#
# ## кол-во объектов == кол-ву ответов
#     assert len(X) == len(y), print(seq([X, y]).map(lambda x: len(x)))
#
# ## сортировка выборки по нужному признаку
#     X, y = zip(*sorted(zip(X, y), key=lambda t: t[0][1]))
#     X, y = map(lambda x: list(x), [X, y])
#     # print(*zip(X,y),sep="\n")
#
# ## пройтись по всем объектам и занести все разбиения на классы
#     X_cuted = X[:,:1]
#     print(X_cuted)
#
#     # F = []
#     # for i in range(5):
#     #     F.append(lambda x: x<10)
#     #
#     # for x in X:
#     #     print(F[0](x))
