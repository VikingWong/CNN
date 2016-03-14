import theano
import numpy as np
from util import Params

def create_theano_func(name, data, x,y, drop, input, output, batch_size, updates=None, dropping=False, prefix=''):
    '''
    Wrapper for creating a theano function.
    Input must be an array and first element in the input array must be the index.

    '''
    set_x, set_y = data.set[name]
    index = input[0]

    #If only output is interesting and not y, then only include x in set_givens.
    set_givens = { x: set_x[index * batch_size: (index + 1) * batch_size],
                   y: set_y[index * batch_size: (index + 1) * batch_size],
                   drop: np.cast['int32'](int(dropping))}


    return theano.function(
        input,
        output,
        updates=updates,
        name=name + prefix,
        givens=set_givens,
        on_unused_input='warn'
    )


def create_profiler_func(data, x,y, drop, input, output_layer, cost, batch_size):
    '''
    Profiler function which retrieve different output values from the network.
    Uses the training set
    '''
    name = 'train'
    output = (output_layer.output, y, cost, output_layer.errors(y))
    return create_theano_func(name, data, x,y,drop,input, output, batch_size)


def create_output_func(dataset, x, y, drop, input, output_layer, batch_size):
    name= 'output'
    #Compability with create_theano_func
    data = Params({'set': {'output': dataset}})
    output = (output_layer.output, y)
    return create_theano_func(name, data, x,y, drop, input, output, batch_size)
