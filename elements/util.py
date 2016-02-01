import theano
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class BaseLayer(object):

    def __init__(self, rng, input, dropout_rate):
        self.stream = RandomStreams()
        self.input = input
        self.rng = rng
        self.dropout_rate = dropout_rate


    def set_bias(self, b, n):
        if b is None:
            self.b = theano.shared(
                value=np.zeros((n,), dtype=theano.config.floatX),
                name='b',
                borrow=True
            )
        else:
            self.b = b


    def set_weight(self, W, W_low, W_high, size):
        if W is None:
            self.W = self.generate_init_weight(W_low, W_high, size)
        else:
            self.W = W


    def dropout(self, X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= self.stream.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X


    def generate_init_weight(self, low, high, size):
        return theano.shared(
            np.asarray(self.rng.uniform(low=low, high=high, size=size), dtype=theano.config.floatX),
            name='W', borrow=True
        )