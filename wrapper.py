import theano


def create_theano_func(name, data, x,y, input, output, batch_size, updates=None):
    '''
    Wrapper for creating a theano function.
    Input must be an array and first element in the input array must be the index.

    '''
    set_x, set_y = data.set[name]
    index = input[0]
    return theano.function(
        input,
        output,
        updates=updates,
        givens={
            x: set_x[index * batch_size: (index + 1) * batch_size],
            y: set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

def create_profiler_func(data, x,y, input, output_layer, cost, batch_size):
    '''
    Profiler function which retrieve different output values from the network.
    Uses the training set
    '''
    name = 'train'
    output = (output_layer.output, y, cost, output_layer.errors(y))
    return create_theano_func(name, data, x,y,input, output, batch_size)
