import theano
import numpy as np
import theano.tensor as T
class Util(object):

    @staticmethod
    def create_weights(n_in, n_out):
        W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        return W

    @staticmethod
    def create_bias(n_out):
        b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        return b

    #Rectified linear unit
    @staticmethod
    def reLU(wb):
        output = T.maximum(0.0, wb)
        return output

    # sigmoid
    @staticmethod
    def sigmoid(wb):
        output = T.nnet.sigmoid(wb)
        return output

    # tanh
    @staticmethod
    def tanh(wb):
        output = T.tanh(wb)
        return(output)

    # softmax
    @staticmethod
    def softmax(x):
        return T.nnet.softmax(x)