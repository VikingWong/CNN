import theano
import theano.tensor as T
import numpy as np
from elements.util import Util

class OutputLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, verbose=True):
        self._verbose_print(verbose, n_in, n_out)
        self.input = input
        self.n = n_out
        #TODO: Clever way to use create_weights method
        if W is None:
            self.W = theano.shared(value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high= np.sqrt(6.0 /(n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ), name='W', borrow=True)
        else:
            print("Using supplied weight matrix")
            self.W = W

        if b is None:
            self.b = Util.create_bias(n_out)
        else:
            print("Using supplied bias'")
            self.b = b

        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        #return T.sum(T.nnet.binary_crossentropy(self.output, y));
        return -T.mean((y * T.log(self.output)) + ( (1 -y ) * T.log(1-self.output) ))

    def errors(self, y):
        #Mean squared error no percentage error at all!
       return T.sum(T.pow(self.output- y, 2))/self.n;

    def _verbose_print(self, is_verbose, n_in, n_out):
        if is_verbose:
            print('Initializing Output layer with', n_out, 'outputs')
            print('')
