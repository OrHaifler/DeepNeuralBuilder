import numpy as np
from sklearn.utils import shuffle
from functions_utils import non_linearities

def gradient_descent(NN, X: np.ndarray, y: np.ndarray, loss_method: str, lr=1e-3, epochs=1000):

    N = X.shape[0]

    for epoch in range(epochs):
        NN.backward(X, y, loss_method)

        for l in range(NN.L - 1, -1, -1):
            if NN.layers[l].type in non_linearities:
                continue
            dW, db = NN.grads[l][1:]
            NN.layers[l].weights['W'] -= lr * dW
            NN.layers[l].weights['b'] -= lr * db
        if epoch % 100 == 0:
            print(NN.loss)

def stochastic_gradient_descent(NN, X: np.ndarray, y: np.ndarray, loss_method: str, lr=1e-3, epochs=1000, batch_size=50):


    N = X.shape[0]
    X_, y_ = shuffle(X, y)
    for epoch in range(epochs):
        X_, y_ = shuffle(X_, y_)
        for (batchX, batchy) in next_batch(X_, y_, batch_size):

            NN.backward(batchX, batchy, loss_method)
            if epoch == 0:
                print(NN.loss)
            for l in range(NN.L - 1, -1, -1):
                if NN.layers[l].type in non_linearities:
                    continue
                dW, db = NN.grads[l][1:]
                NN.layers[l].weights['W'] -= lr * dW
                NN.layers[l].weights['b'] -= lr * db
        if epoch % 100 == 0:

            print(NN.loss)


def momentum_GD(NN, X: np.ndarray, y: np.ndarray, loss_method: str, lr=1e-3, gamma=1, epochs=1000):

    N = X.shape[0]
    v = [0 for i in range(NN.L)]
    for l in range(NN.L - 1, -1, -1):
        if NN.layers[l].type not in non_linearities:
            v[l] = [np.zeros(NN.layers[l].weights['W'].shape), np.zeros(NN.layers[l].weights['b'].shape)]

    for epoch in range(epochs):
        NN.backward(X, y, loss_method)

        for l in range(NN.L - 1, -1, -1):
            if NN.layers[l].type in non_linearities:
                continue
            dW, db = NN.grads[l][1:]
            if np.linalg.norm(dW) > 1000:
                dW *= 1000 / np.linalg.norm(dW)
            if np.linalg.norm(db) > 1000:
                db *= 1000 / np.linalg.norm(db)

            momentum_updateW = gamma * v[l][0]
            momentum_updateb = gamma * v[l][1]
            NN.layers[l].weights['W'] -= lr * (dW + momentum_updateW)
            NN.layers[l].weights['b'] -= lr * (db + momentum_updateb)
            v[l][0], v[l][1] = dW, db
        if epoch % 100 == 0:
            print(NN.loss)


def adagrad(NN, X: np.ndarray, y: np.ndarray, loss_method: str, e=1e-6, n=0.013, epochs=100):

    N = X.shape[0]
    G = [0 for i in range(NN.L)]
    for l in range(NN.L - 1, -1, -1):
        if NN.layers[l].type not in non_linearities:
            G[l] = [np.zeros(NN.layers[l].weights['W'].shape), np.zeros(NN.layers[l].weights['b'].shape)]

    for epoch in range(epochs):
        NN.backward(X, y, loss_method)

        for l in range(NN.L - 1, -1, -1):
            if NN.layers[l].type in non_linearities:
                continue
            dW, db = NN.grads[l][1:]
            if np.linalg.norm(dW) > 1000:
                dW *= 1000 / np.linalg.norm(dW)
            if np.linalg.norm(db) > 1000:
                db *= 1000 / np.linalg.norm(db)

            G[l][0] += dW ** 2
            G[l][1] += db ** 2
            NN.layers[l].weights['W'] -= n / (np.sqrt(G[l][0] + e)) * dW
            NN.layers[l].weights['b'] -= n / (np.sqrt(G[l][1] + e)) * db

        if epoch % 100 == 0:
            print(NN.loss)

def RMSProp(NN, X: np.ndarray, y: np.ndarray, loss_method: str, e=1e-3, gamma=0.9, n=0.013, epochs=100):

    N = X.shape[0]
    s = [0 for i in range(NN.L)]
    for l in range(NN.L - 1, -1, -1):
        if NN.layers[l].type not in non_linearities:
            s[l] = [np.zeros(NN.layers[l].weights['W'].shape), np.zeros(NN.layers[l].weights['b'].shape)]

    for epoch in range(epochs):
        NN.backward(X, y, loss_method)

        for l in range(NN.L - 1, -1, -1):
            if NN.layers[l].type in non_linearities:
                continue
            dW, db = NN.grads[l][1:]
            if np.linalg.norm(dW) > 1000:
                dW *= 1000 / np.linalg.norm(dW)
            if np.linalg.norm(db) > 1000:
                db *= 1000 / np.linalg.norm(db)

            s[l][0] = gamma * s[l][0] + (1 - gamma) * dW ** 2
            s[l][1] = gamma * s[l][1] + (1 - gamma) * db ** 2
            NN.layers[l].weights['W'] -= n / (np.sqrt(s[l][0] + e)) * dW
            NN.layers[l].weights['b'] -= n / (np.sqrt(s[l][1] + e)) * db

        if epoch % 100 == 0:
            print(NN.loss)


def next_batch(X, y, batch_size):

    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])
