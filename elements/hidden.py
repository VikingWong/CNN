import theano
import theano.tensor as T
import numpy as np
from elements.util import BaseLayer

class HiddenLayer(BaseLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, verbose=True, dropout_rate=0.0):
        super().__init__(rng, input, dropout_rate)
        self._verbose_print(verbose,activation, n_in, n_out)

        W_bound = np.sqrt(6.0 / (n_in + n_out)) * 4
        self.set_weight(W, -W_bound, W_bound, (n_in, n_out))
        self.set_bias(b, n_out)

        lin_output = T.dot(input, self.W) + self.b
        lin_output = self.dropout(lin_output, dropout_rate)
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
