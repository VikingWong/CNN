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
                W.values *= 4

            W = theano.shared(value=w_values, name='W', borrow=True)
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

    def _verbose_print(self, is_verbose,activation, n_in, n_out):
        #Add type of activation
        if is_verbose:
            print('Initializing hidden layer with', n_out, 'nodes, where each have', n_in, 'incoming connections')
            print('')
