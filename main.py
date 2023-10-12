import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from optimizer import *
from initializer import initialize
from layers import *
from functions_utils import *
from loss import *
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from pandas import DataFrame


class Layer:
    def __init__(self, layer, dropout=0):
        self.dropout = dropout
        self.weights = {}
        if layer['type'] not in non_linearities:
            self.weights['W'], self.weights['b'] = initialize(layer)
        self.forward, self.backward = function_utils[layer['type']]
        self.type = layer['type']



class NeuralNetwork:
    def __init__(self, layer_array):
        self.types = ['fc', 'conv', 'batchnorm', 'max_pooling',
                      'average_pooling', 'softmax', 'relu', 'sigmoid', 'tanh']
        self.L = len(layer_array)
        self.layers = [Layer(layer_array[i]) for i in range(self.L)]
        self.caches = {}
        self.grads = {}
        self.loss = 0

    def forward(self, X: np.ndarray, y: np.ndarray, loss_method: str) -> (np.ndarray, float):
        out = X
        for i in range(self.L):
            layer = self.layers[i]
            if layer.type in non_linearities:
                out, self.caches[i] = layer.forward(out)
            else:
                out, self.caches[i] = layer.forward(out, layer.weights['W'], layer.weights['b'], layer.dropout)
        return loss_method(out, y)

    def backward(self, X: np.ndarray, y: np.ndarray, loss_method: str):

        self.loss, self.grads[self.L] = self.forward(X, y, loss_method)
        self.grads[self.L] = [self.grads[self.L]]
        for l in range(self.L - 1, -1, -1):
            if self.layers[l].type in non_linearities:
                self.grads[l] = self.layers[l].backward(self.grads[l + 1][0], self.caches[l])
                continue
            self.grads[l] = self.layers[l].backward(self.loss, self.grads[l + 1][0], self.caches[l])
        Z = self.predict_arr(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        df = DataFrame(dict(x=T[:, 0], y=T[:, 1], label=R))
        colors = {0: 'blue', 1: 'orange'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

        # Label the axes
        plt.xlabel('X_1')
        plt.ylabel('X_2')

        # Show the plot
        plt.show()

    def predict(self, X):

        out = X
        for i in range(self.L):
            layer = self.layers[i]
            if layer.type in non_linearities:
                out, _ = layer.forward(out)
            else:
                out, _ = layer.forward(out, layer.weights['W'], layer.weights['b'], layer.dropout)
        return out[0].argmax()


    def predict_arr(self, X):

        pred = np.zeros(X.shape[0])
        for l in range(len(X)):
            pred[l] = self.predict(X[l])
        return pred

    def train(self, X, y, loss_method, lr=1e-3, clipping_threshold=10000, epochs=1000):

        N = X.shape[0]
        self.history = []
        stochastic_gradient_descent(self, X, y, loss_method, lr, epochs)





# Test example


#X = np.random.randint(0, 10, size=(50, 4)).astype(float)
#y = X.sum(axis = 1)


X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=5, random_state=11)
T, R = X, y
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'blue', 1:'orange'}
fig, ax = plt.subplots()
grouped = df.groupby('label')

for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()

layers_arr = [{'type': 'fc', 'dim': (2, 4), 'ext': (1, ), 'init_method': 'random_normal', 'dropout': 0},
              {'type': 'fc', 'dim': (4, 2), 'ext': (1, ), 'init_method': 'random_normal', 'dropout': 0},
              {'type': 'sigmoid', 'dim': None, 'ext': (1, ), 'init_method': None, 'dropout': 0}]
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Network = NeuralNetwork(layers_arr)
Network.train(X, y, binary_crossentropy, epochs=1000)







'''np.random.seed(25)
class1_mean, class1_cov = [2,2], [[1, 0.5], [0.5, 1]]
class2_mean, class2_cov = [6,6], [[1, -0.5], [-0.5, 1]]
class1_data = np.random.multivariate_normal(class1_mean, class1_cov, 25)
class2_data = np.random.multivariate_normal(class2_mean, class2_cov, 25)
#class1_data = np.array([[1,3], [2,4], [4,5]])
#class2_data = np.array([[6,7], [7,6], [8,5]])
plt.scatter(class1_data[:, 0], class1_data[:, 1], c='b', label='Class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1], c='r', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.show()

X = np.concatenate((class1_data, class2_data), axis = 0)
y = np.concatenate((np.zeros((25)), np.ones((25))), axis = 0).astype(int)'''
