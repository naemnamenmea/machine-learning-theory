from pprint import pprint
import numpy as np

predicates = [
    lambda x: x < 10,
    lambda x: x < 5,
    lambda x: x > 2,
    lambda x: x > 7
]


def Informativity(p):
    return p


class Node:
    def __init__(self):
        self.left_leaf = None,
        self.right_leaf = None


class Tree:
    def __init__(self):
        pass


def division(X, y, predicates):
    best_pred = None
    best_score = 0
    length = len(y)
    total = np.append(y,np.zeros(length),axis=1)
    for cl in y:
        total[cl] += 1
    for pred in predicates:
        score = []
        for x, cl in zip(X, y):
            if (pred(x) == cl):
                score[cl] += 1
        total_score = 0
        for x, cnt in score:
            total_score += Informativity(cnt/total[x])
        if best_score < total_score:
            best_score = total_score
            best_pred = pred

    return best_pred

if __name__ == '__main__':
    dataset = [
        ([1], False),
        ([2], True),
        ([3], False),
        ([4], False),
        ([5], False),
        ([6], True),
        ([7], False),
        ([8], True),
        ([9], True),
        ([10], True),
        ([11], False)
    ]

    print(predicates)
    best_pred = division(dataset[:, 0], dataset[:, 1], predicates)
    print(best_pred)
