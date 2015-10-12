import theano
import theano.tensor as T
import numpy as np
from elements.util import Util

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, verbose=True):
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
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        #TODO: Wrong name for it.
        return T.mean(T.nnet.binary_crossentropy(self.output, y))

    def errors(self, y):
        #TODO: only copy pasted!
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.output, y))
        else:
            raise NotImplementedError()

    def _verbose_print(self, is_verbose,activation, n_in, n_out):
        #Add type of activation
        if is_verbose:
            print('Initializing hidden layer with', n_out, 'nodes, where each have', n_in, 'incoming connections')
            print('')
