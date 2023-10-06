import numpy as np

def numerical_gradient(f, x, verbose=True, h=0.00001):

    grad = np.zeros_like(x)
    fx = f(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index
        tmp = x[ix]
        x[ix] = tmp + h
        fxh1 = f(x)
        x[ix] = tmp - h
        fxh2 = f(x)
        x[ix] = tmp

        print((fxh1 - fxh2) / (2 * h))
        grad[ix] = (fxh1 - fxh2) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad


X = np.array([1,2,3])
f = lambda X: X.mean()
print(eval_numerical_gradient(f, X))
