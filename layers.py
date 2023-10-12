import numpy as np

def fc_forward(X, W, b , p=0):
    #V
    mask = np.random.rand(*X.shape) < 1 - p
    X *= mask
    out = X.dot(W) + b
    out *= 1 / (1 - p)
    cache = (X, W, b, mask)
    return out, cache


def fc_backward(out, dstream, cache):  # cache = (X,W,b, mask)
    #V
    X, W, b, mask = cache
    dX = (dstream).dot(W.T) * mask
    dW = (X.T).dot(dstream)
    db = dstream.sum(axis=0)

    return (dX, dW, db)


def conv_forward(X, kernel):
    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), 'constant')
    if len(kernel.shape) > 2:
        C = kernel.shape[0]
        if (image.shape[0] != C):
            return "Input channels error"
    Kx, Ky = kernel.shape[1:]
    Ix, Iy = image.shape[1:]

    outX = 2 * padding + Ix - Kx + 1
    outY = 2 * padding + Iy - Ky + 1
    output = np.zeros((C, outX, outY))

    for i in range(outX):
        for j in range(outY):
            for c in range(C):
                output[c, i, j] = (padded[c, i:i + Kx, j:j + Ky] * kernel[c]).sum()
    return output, cache


def conv_backward(out, dstream, cache):
    pass


# Remember scaling to maintain expectation
def dropout_forward(X, p):
    #V
    mask = np.random.rand(*X.shape) < 1 - p
    out = X * mask
    return out, mask


def dropout_backward(out, dstream, cache):  # cache = (X, mask)
    #V
    mask = cache[1]
    dstream *= mask
    return dstream


def batchnorm_forward(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=0)
    xmu = X - mu
    sq = xmu ** 2
    var = sq.mean(axis=0)
    rt = np.sqrt(var + eps)
    inv = 1 / rt
    mul = xmu * inv
    xg = gamma * xmu
    out = xg + beta
    cache = (eps, mu, xmu, sq, var, rt, inv, mul, xg)


def batchnorm_backward(out, dout, cache):
    (eps, mu, xmu, sq, var, rt, inv, mul, xg) = cache

    dbeta = dout.sum(axis=0)
    dxg = dout
    dgamma = (dxg * mul).sum(axis=0)
    dmul = dxg * gamma
    dinv = (dmul * xmu).sum(axis=0)
    dxmu1 = dmul * inv
    drt = dinv * (-1 / rt ** 2)
    dvar = drt * (1 / np.sqrt(var + eps))
    dsq = np.ones(out.shape) * 1 / N * dvar
    dxmu2 = 2 * dsq * xmu
    dxmu = dxmu1 + dxmu2
    dx1 = dxmu  # dx1 = np.ones(out.shape) * dxmu = dxmu
    dmu = -1 * dxmu.sum(axis=0)
    dx2 = 1 / N * np.ones(out.shape) * dmu
    dx = dx1 + dx2

    return (dbeta, dxg, dgamma, dmul, dinv, dxmu1, drt, dvar, dsq, dxmu2, dxmu, dx1, dmu, dx2, dx)


def max_pooling_forward(X, spatial=2, stride=2):
    C, Xw, Xh = X.shape
    Ow = (Xw - spatial) / stride + 1
    Oh = (Xh - spatial) / stride + 1
    args = []
    out = np.zeros((int(Ow), int(Oh)))
    for i in range(0, Xw - spatial + 1, stride):
        for j in range(0, Xh - spatial + 1, stride):
            submatrix = X[i:i + spatial, j:j + spatial]
            out[int(i / stride), int(j / stride)] = submatrix.max()
            argmax_submatrix = np.unravel_index(np.argmax(submatrix), submatrix.shape)
            args.append((argmax_submatrix[0] + i, argmax_submatrix[1] + j, i, j))
    return out


def max_pooling_backward(dout, cache):
    X, args = cache
    dX = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (i, j) in args:
                dX[i, j] = X[i, j]


def average_pooling_forward(X, spatial=2, stride=2):
    C, Xw, Xh = X.shape
    Ow = (Xw - spatial) / stride + 1
    Oh = (Xh - spatial) / stride + 1
    args = []
    out = np.zeros((C, int(Ow), int(Oh)))
    for i in range(0, Xw - spatial + 1, stride):
        for j in range(0, Xh - spatial + 1, stride):
            for c in range(C):
                out[c, int(i / stride), int(j / stride)] = X[c, i:i + spatial, j:j + spatial].mean()

    return out


def average_pooling_backward(dout, cache):
    X = cache
    dX = np.zeros(X.shape)

    for c in range(X.shape[0]):
        for i in range(0, Xw - spatial + 1, stride):
            for j in range(0, Xh - spatial + 1, stride):
                dX[c, i:i + spatial, j:j + spatial] = out[c, int(i / stride), int(j / stride)]
    dX *= 1 / N
    return dX

def ID_forward(X1, X2=None):

    return X1

def ID_backward(X1, X2=None):

    return X1
def softmax_forward(X, y):

    N = X.shape[0]
    Xn = X - X.max(axis = 1).reshape(N,1)
    axis_sum = (np.exp(Xn)).sum(axis = 1)
    out = np.exp(Xn[arange(N), y]) / axis_sum
    return out

def softmax_backward():

    pass


def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache


def relu_backward(dstream, cache):  # cache = (X)

    X = cache
    dstream *= np.heaviside(X, 0)  # np.ones(X.shape) * (X > 0).astype(int)
    return [dstream]


def sigmoid_forward(X):

    out = 1 / (1 + np.exp(-X))
    cache = X, out
    return out, cache


def sigmoid_backward(dstream, cache):  # cache = (X, out)

    X, out = cache
    ds = out * (1 - out)
    dX = dstream * ds

    return [dX]


def tanh_forward(X):

    out = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    cache = X, out
    return out, cache


def tanh_backward(dstream, cache):  # cache = (X)

    X, out = cache
    ds = 1 - out ** 2
    dX = dstream * ds

    return [dX]
