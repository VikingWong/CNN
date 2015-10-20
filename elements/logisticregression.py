import theano
import theano.tensor as T
import numpy as np
from elements.util import BaseLayer

class LogisticRegression(BaseLayer):

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """
        super().__init__(None, input)
        if W is None:
            self.W  = theano.shared(
                value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        self.set_bias(b, n_out)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        #TODO: Insert softmax again.
        #self.p_y_given_x = Util.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.probabilities = T.log(self.p_y_given_x)

    def negative_log_likelihood(self, y):
        #TODO: adapt cost function to 2D data.
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