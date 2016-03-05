import theano
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

class BaseLayer(object):

    def __init__(self, rng, input, dropout_rate):
        self.input = input
        self.rng = rng
        #self.srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.srng = RandomStreams(int(time.time()))
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


    def dropout(self, X, p=1.0):
        #The probability of NOT dropping out a unit
        if p == 1:
            #No dropout when p = 1. keep all.
            return X
        if p == 0:
            raise Exception("Can't drop everything!")

        mask = self.srng.binomial(n=1, p=p, size=X.shape, dtype=theano.config.floatX)
        output =  X * mask
        return  np.cast[theano.config.floatX](1.0/p) * output


    def generate_init_weight(self, low, high, size):
        return theano.shared(
            np.asarray(self.rng.uniform(low=low, high=high, size=size), dtype=theano.config.floatX),
            name='W', borrow=True
        )