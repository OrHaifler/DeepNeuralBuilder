import numpy as np


class NN:
    #Initialize a neural network
    def __init__(self, layers, init_method):
        self.params = {}
        self.L = len(layers)

        for i in range(L):
                    #Write an init_method that will initialize parameters
                    self.params[f'W{i}'] = initialize(layers, "W", init_method)
                    self.params[f'b{i}'] = initialize(layers, "b", init_method)
                    self.params[f'p{i}'] = initialize(layers, "p", init_method)

    #Define forward method for each layer type

    def fc_forward(self, X, W, b):

        out = X.dot(W) + b
        return out

    def fc_backward(self, out, dstream, cache): #cache = (X,W,b)

        dW = (X.T).dot(dstream)
        dX = dstream.dot(W.T)
        db = dstream.sum(axis = 0)

        return (dX, dW, db)

    def conv_forward(self, X, kernel):

        pass

    def conv_backward(self, out, dstream, cache):

        pass

    #Remember scaling to maintain expectation
    def dropout_forward(self, X, p):

        mask = np.random.rand(*X.shape) < p
        out = X * mask
        return out, mask

    def dropout_backward(self, out, dstream, cache): #cache = (X, mask)

        mask = cache[1]
        dstream *= mask
        return dstream

    def bachnorm_forward(self, X, gamma, beta, eps = 1e5):

        mu = X.mean(axis = 0)
        xmu = X - mu
        sq = xmu**2
        var = sq.mean(axis = 0)
        rt = np.sqrt(var + eps)
        inv = 1/rt
        mul = xmu * inv
        xg = gamma * xmu
        out = xg + beta
        cache = (eps, mu, xmu, sq, var, rt, inv, mul, xg)
    def batchnorm_backward(self, out, dout, cache):

        (eps, mu, xmu, sq, var, rt, inv, mul, xg) = cache

        dbeta = dout.sum(axis = 0)
        dxg = dout
        dgamma = (dxg * mul).sum(axis=0)
        dmul = dxg * gamma
        dinv = (dmul * xmu).sum(axis = 0)
        dxmu1 = dmul * inv
        drt = dinv * (-1 / rt**2)
        dvar = drt * (1 / np.sqrt(var + eps))
        dsq = np.ones(out.shape) * 1/N * dvar
        dxmu2 = 2 * dsq * xmu
        dxmu = dxmu1 + dxmu2
        dx1 = dxmu # dx1 = np.ones(out.shape) * dxmu = dxmu
        dmu = -1 * dxmu.sum(axis = 0)
        dx2 = 1 / N * np.ones(out.shape) * dmu
        dx = dx1 + dx2

        return (dbeta, dxg, dgamma, dmul, dinv, dxmu1, drt, dvar, dsq, dxmu2, dxmu, dx1, dmu, dx2, dx)




    def relu_forward(self, X):

        out = np.maximum(X, 0)
        return out

    def relu_backward(self, out, dstream, cache): #cache = (X)

        dstream *= np.heaviside(out, 0) #np.ones(X.shape) * (X > 0).astype(int)
        return dstream

    def sigmoid_forward(self, X):

        out = 1 / (1 + np.exp(-X))
        return out


    def sigmoid_backward(self, out, dstream, cache): #cache = (X)

        ds = out * (1 - out)
        dX = dstream * ds

        return dX

    def tanh_forward(self, X):

        out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return out

    def tanh_backward(self, out, dstream, cache): #cache = (X)

        ds = 1 - out ** 2
        dX = dstream * ds

        return dX



















