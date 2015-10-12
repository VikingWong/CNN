import theano
import theano.tensor as T
import numpy as np
from elements.util import Util

class OutputLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid, verbose=True):
        self._verbose_print(verbose,activation, n_in, n_out)
        self.input = input
        #TODO: Clever way to use create_weights method
        if W is None:
            w_values =np.asarray(
                rng.uniform(
                    low=-np.sqrt(6.0 / (n_in + n_out)),
                    high= np.sqrt(6.0 /(n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                w_values = w_values * 4

            self.W = theano.shared(value=w_values, name='W', borrow=True)
        else:
            self.W = W

        if b is None:
            self.b = Util.create_bias(n_out)
        else:
            self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.y_pred = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return T.nnet.binary_crossentropy(y, self.y_pred);

    def errors(self, y):
       return T.pow(2, T.sub(self.y_pred, y));

    def _verbose_print(self, is_verbose,activation, n_in, n_out):
        if is_verbose:
            print('Initializing Output layer with', n_out, 'outputs')
            print('')
