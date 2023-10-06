import numpy as np
from initializer import initialize
from layers import *
class NN:
    #Initialize a neural network
    def __init__(self, layers):
        self.params = {}
        #self.init_functions = {"fc": "fc_initialize"}
        self.L = len(layers)

        for i in range(self.L):
                    #Write an init_method that will initialize parameters
                    self.params[f'W{i}'], self.params[f'b{i}']  = initialize(layers[i])


    #Define forward method for each layer type





X = np.random.randint(5, size = (3,3))
layers = [{'dim': (3,2), 'ext': (1, ), 'init_method': 'random_normal'}, {'dim': (2,1), 'ext': (1, ), 'init_method': 'random_normal'}]
Test = NN(layers)
W1, b1 = Test.params['W0'], Test.params['b0']
W2, b2 = Test.params['W1'], Test.params['b1']
Z1 = fc_forward(X, W1, b1)
Z2 = fc_forward(Z1, W2, b2)
print(Z2)



















