import numpy as np


def cross_entropy(yh, y):
    #yh: (20,4)
    #y: (20,1)
    N = yh.shape[0]
    yhn = np.exp(yh - yh.max(axis = 1).reshape(N, 1))
    scores = np.exp(yhn) / np.exp(yhn).sum(axis = 1).reshape(N,1)
    loss = -(np.log(scores[np.arange(N), y])).mean()
    scores[np.arange(N), y] -= 1
    dX = scores / N
    return loss, dX

def svm(yh, y, delta=1):

    N = yh.shape[0]
    margins = np.maximum(0, X - X[np.arange(N), y].reshape(N,1) + delta)
    margins[np.arange(N), y] = 0
    loss = margins.sum()
    dX = (margins > 0).astype(int)
    return loss, dX



def MSE(yh, y):

    N = yh.shape[0]
    diff = yh - y.reshape(N,1)
    loss = (diff ** 2).sum() / (2 * N)
    dX = 1 / N * diff
    return loss, dX

