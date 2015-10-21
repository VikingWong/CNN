import theano
import theano.tensor as T
import numpy as np
from elements.util import BaseLayer

class OutputLayer(BaseLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, verbose=True):
        super().__init__(rng, input, 0.0)
        self._verbose_print(verbose, n_in, n_out)
        self.output_dim = n_out

        W_bound = np.sqrt(6.0 / (n_in + n_out)) * 4
        self.set_weight(W, -W_bound, W_bound, (n_in, n_out))
        self.set_bias(b, n_out)

        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        #return T.sum(T.nnet.binary_crossentropy(self.output, y));
        return -T.mean((y * T.log(self.output)) + ( (1 -y ) * T.log(1-self.output) ))

    def errors(self, y):
        #Mean squared error no percentage error at all!
       return (T.sum(T.pow(self.output- y, 2))/self.output_dim);

    def _verbose_print(self, is_verbose, n_in, n_out):
        if is_verbose:
            print('Initializing Output layer with', n_out, 'outputs')
            print('')
