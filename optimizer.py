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
            if np.linalg.norm(dW) > 1000:
                dW *= 1000 / np.linalg.norm(dW)
            if np.linalg.norm(db) > 1000:
                db *= 1000 / np.linalg.norm(db)
            NN.layers[l].weights['W'] -= lr * dW / N
            NN.layers[l].weights['b'] -= lr * db / N
        print(NN.loss)

def stochastic_gradient_descent(NN, X: np.ndarray, y: np.ndarray, loss_method: str, lr=1e-3, epochs=1000, batch_size=10):


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
                '''if np.linalg.norm(dW) > 1000:
                    dW *= 1000 / np.linalg.norm(dW)
                if np.linalg.norm(db) > 1000:
                    db *= 1000 / np.linalg.norm(db)'''
                NN.layers[l].weights['W'] -= lr * dW
                NN.layers[l].weights['b'] -= lr * db
        print(NN.loss)



def next_batch(X, y, batch_size):

    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])








