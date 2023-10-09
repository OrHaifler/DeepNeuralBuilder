import numpy as np
import matplotlib.pyplot as plt
from initializer import initialize
from layers import *
from functions_utils import *
from loss import *


class Layer:

    def __init__(self, layer):
        self.weights = {}
        self.weights['W'], self.weights['b'] = initialize(layer)
        self.forward, self.backward = function_utils[layer['type']]

class NN:
    #Initialize a neural network
    def __init__(self, layer_array):
        self.types = ['fc', 'conv', 'dropout', 'batchnorm', 'max_pooling',
                      'average_pooling', 'softmax', 'relu', 'sigmoid', 'tanh']
        self.L = len(layer_array)
        self.layers = []
        for i in range(self.L):
            self.layers.append(Layer(layer_array[i]))
        self.caches = {}
        self.grads = {}


    def single_forward(self, X: np.ndarray, layer: Layer) -> (np.ndarray, tuple):

        out, cache = layer.forward(X, layer.weights['W'], layer.weights['b'])
        return out, cache

    def forward(self, X, y, loss_method):

        out = X
        for i in range(self.L):
            out, self.caches[i] = self.single_forward(out, self.layers[i])
        return loss_method(out, y)

    def predict(self, X):

        out = X
        for i in range(self.L):
            out, self.caches[i] = self.single_forward(out, self.layers[i])
        return out



    def train(self, X, y, loss_method, lr=1e-3, clipping_threshold=10000,  epochs=100000):

        N = X.shape[0]
        self.history = []

        for epoch in range(epochs):

            loss, self.grads[self.L] = self.forward(X, y, loss_method)
            self.grads[self.L] = [self.grads[self.L]]
            for l in range(self.L - 1, -1, -1):
                self.grads[l] = fc_backward(loss, self.grads[l + 1][0], self.caches[l])
                dX, dW, db = self.grads[l]
                dW_norm, db_norm = np.linalg.norm(dW), np.linalg.norm(db)
                if dW_norm > clipping_threshold:
                    dW *= clipping_threshold / dW_norm
                if db_norm > clipping_threshold:
                    db *= clipping_threshold / db_norm
                self.layers[l].weights['W'] -= lr * dW / N
                self.layers[l].weights['b'] -= lr * db / N
            if epoch % 1000 == 0:
                self.history.append(loss)
                print(epoch, loss)
                print("--------------")
        plt.plot(self.history)
        plt.show()





#Test example
X = np.random.randint(0, 10, size=(50,4))
y = X.sum(axis = 1)
layers_arr = [{'type': 'fc', 'dim': (4,64), 'ext': (1, ), 'init_method': 'random_normal'},
            {'type': 'fc', 'dim': (64,32), 'ext': (1, ), 'init_method': 'random_normal'},
              {'type': 'fc', 'dim': (32,1), 'ext': (1, ), 'init_method': 'random_normal'}]
Network = NN(layers_arr)
Network.train(X, y, MSE, epochs=10000)
print(Network.predict(np.array([1, 5, 20, 3])))




'''dX2, dW2, db2 = fc_backward(loss, dloss, self.caches[1])
dX1, dW1, db1 = fc_backward(loss, dX2, self.caches[0])
self.layers[1].weights['W'] -= lr * dW2 / X.shape[0]
self.layers[1].weights['b'] -= lr * db2 / X.shape[0]
self.layers[0].weights['W'] -= lr * dW1 / X.shape[0]
self.layers[0].weights['b'] -= lr * db1 / X.shape[0]
self.history.append((epoch, loss))
if epoch % 1000 == 0:
    print(epoch, loss)
    print("--------------")'''







