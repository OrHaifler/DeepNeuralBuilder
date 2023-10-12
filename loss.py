import numpy as np


def cross_entropy(yh, y):
    #yh: (20,4)
    #y: (20,1)
    N = yh.shape[0]
    #yhn = np.exp(yh - yh.max(axis = 1).reshape(N, 1))
    scores = np.exp(yh) / np.exp(yh).sum(axis = 1).reshape(N,1)
    loss = -(np.log(scores[np.arange(N), y.reshape(N,1)])).mean()
    dX = scores.copy()
    dX[np.arange(N), y] -= 1
    dX /= N
    return loss, dX

def binary_crossentropy(y_pred, y_true):
    epsilon = 1e-15  # Small constant to avoid division by zero

    # Ensure y_pred is in the range [epsilon, 1-epsilon]
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute the binary cross-entropy loss for each sample in the batch
    loss = -(y_true * np.log(y_pred[:, 1]) + (1 - y_true) * np.log(y_pred[:, 0]))

    # Compute the gradient of the loss with respect to y_pred
    grad = np.zeros_like(y_pred)
    grad[:, 1] = (y_pred[:, 1] - y_true) / (y_pred[:, 1] * (1 - y_pred[:, 1]))
    grad[:, 0] = (y_pred[:, 0] - (1 - y_true)) / ((1 - y_pred[:, 0]) * y_pred[:, 0])

    return loss.mean(), grad



def svm(yh, y, delta=1):

    N = yh.shape[0]
    margins = np.maximum(0, yh - yh[np.arange(N), y].reshape(N,1) + delta)
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

