import theano
import theano.tensor as T
import numpy as np
from elements.util import Util

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """

        if W is None:
            self.W = Util.create_weights(n_in, n_out)
        else:
            self.W = W

        if b is None:
            self.b = Util.create_bias(n_out)
        else:
            self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        self.p_y_given_x = Util.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        self.probabilities = T.log(self.p_y_given_x)

    def negative_log_likelihood(self, y):
        print(y)
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()