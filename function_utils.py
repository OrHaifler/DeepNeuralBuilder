from layers import *


#Fix this with: function_utils = {type : (eval(type + "_forward), eval(type + "_backward")) for type in types}
function_utils = {
    'fc': (fc_forward, fc_backward),
    'conv': (fc_forward, fc_backward),
    'batchnorm': (batchnorm_forward, batchnorm_backward),
    'max_pooling': (max_pooling_forward, max_pooling_backward),
    'average_pooling': (average_pooling_forward, average_pooling_backward),
    'softmax': (softmax_forward, softmax_backward),
    'relu': (relu_forward, relu_backward),
    'sigmoid': (sigmoid_forward, sigmoid_backward),
    'tanh': (tanh_forward, tanh_backward),
    None: (ID_forward, ID_backward)
                  }

types = ['fc', 'conv', 'dropout', 'batchnorm', 'max_pooling',
                      'average_pooling', 'softmax', 'relu', 'sigmoid', 'tanh']
non_linearities = ['relu', 'softmax', 'tanh', 'sigmoid']
function_utils = {type : (eval(type + "_forward"), eval(type + "_backward")) for type in types}
function_utils[None] = (ID_forward, ID_backward)
