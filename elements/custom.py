import theano
import theano.tensor as T
import numpy as np
from elements.util import BaseLayer

class OutputLayer(BaseLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, verbose=True):
        super(OutputLayer, self).__init__(rng, input, 0.0)
        self._verbose_print(verbose, n_in, n_out)

        self.negative_log_likelihood = self.crossentropy

        W_bound = np.sqrt(6.0 / (n_in + n_out)) * 4
        self.set_weight(W, -W_bound, W_bound, (n_in, n_out))
        self.set_bias(b, n_out)

        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.output = T.clip(self.output, 1e-7, 1.0 - 1e-7)

        self.params = [self.W, self.b]
        self.input = input


    def loss_crossentropy(self, y):
        return T.mean(T.nnet.binary_crossentropy(self.output, y))

    def loss_bootstrapping(self, y):
        #Customized categorical cross entropy.
        #Based on the multibox impl.
        factor = 0.8
        #TODO: Does this work? Integer and matrix? TEST
        hard = T.gt(self.output, 0.5)
        loss = (
            - T.sum( (factor * y) + ((1- factor) * hard) * T.log(self.output) ) -
            T.sum( (factor * (1 - y)) + ((1- factor) * (1 - hard)) * T.log(1 - self.output) )
        )
        return loss

    def errors(self, y):
        #Returns the mean squared error.
        # Prediction - label squared, for all cells in all batches and pixels.
        # Averaged by sum + divided by total number of elements. AKA - batch_size * dim * dim elements
       return T.mean(T.pow(self.output- y, 2))

    def _verbose_print(self, is_verbose, n_in, n_out):
        if is_verbose:
            print('Output layer with {} outputs'.format(n_out))
            print('---- Sigmoidal units')
            print('')
