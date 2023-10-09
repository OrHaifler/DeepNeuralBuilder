import numpy as np

def initialize(layer):

    init = layer['init_method']
    dim = layer['dim']
    ext = layer['ext']
    init_function = f'{init}_initialize'
    params = eval(init_function + f'{dim, *ext}')
    return params




def random_uniform_initialize(dim, a=0, b=1):

    out = ((b - a) * np.random.rand(*dim) + np.full(dim, a), (b - a) *
           np.random.rand(1,dim[-1]) + np.full((1,dim[-1]), a))
    return out

def random_normal_initialize(dim, scale=1):
    out = (np.random.randn(*dim) * scale, np.random.randn(1, dim[-1]) * scale)
    return out

def constant_initialize(dim, constant=0):
    out = (np.full(dim, constant), np.full((1, dim[-1]), constant))
    return out

def xavier_initialize(dims):

    pass

def he_initialize(dims):

    pass

