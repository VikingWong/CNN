import theano
import theano.tensor as T
import numpy as np
from elements.util import BaseLayer

class HiddenLayer(BaseLayer):
    def __init__(self, rng, input, n_in, n_out, drop, W=None, b=None, activation=T.tanh, verbose=True, dropout_rate=1.0):
        super(HiddenLayer, self).__init__(rng, input, dropout_rate)
        self._verbose_print(verbose,activation, n_in, n_out, dropout_rate)

        W_bound = np.sqrt(6.0 / (n_in + n_out)) * 4
        self.set_weight(W, -W_bound, W_bound, (n_in, n_out))
        self.set_bias(b, n_out)

        lin_output = T.dot(input, self.W) + self.b

        output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        droppedOutput = self.dropout(output, dropout_rate)
        self.output = T.switch(T.neq(drop, 0), droppedOutput, output)
        self.params = [self.W, self.b]


    def _verbose_print(self, is_verbose,activation, n_in, n_out, dropout_rate):
        #Add type of activation
        if is_verbose:
            print('Hidden layer with {} nodes'.format(n_out))
            print('---- Incoming connections: {}'.format(n_in))
            print('---- Dropout rate: {}'.format(dropout_rate))
            if (activation is T.tanh):
                print('---- Activation: tanh')
            elif (activation is T.nnet.relu):
                print('---- Activation: relu')
            print('')