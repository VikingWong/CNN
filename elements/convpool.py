import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
from elements.util import Util

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2),
                 activation=T.tanh, W = None, b = None, verbose = True):
        '''
        :param rng: random number generator used to initialize weights
        :param input: symbolic image tensor
        :param filter_shape:  (number of filters, num input feature maps, filter height, filter width)
        :param image_shape: (batch size, num input feature maps, image height, image width)
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        :param activation:
        :param W:
        :param b:
        :param verbose:
        :return:
        '''
        assert image_shape[1] == filter_shape[1]
        self._verbose_print(verbose, filter_shape, poolsize, image_shape)
        self.input = input
        datatype = theano.config.floatX

        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        if W is None:
            self.W = theano.shared(
                np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=datatype),
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            self.b = Util.create_bias(filter_shape[0])
        else:
            self.b = b

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input

    def _verbose_print(self, is_verbose, filter_shape, poolsize, image_shape):
        if is_verbose:
            print('Initializing convolutional layer with ', filter_shape[0], ' kernels')
            print('----Kernel size ', filter_shape[2], ' X ', filter_shape[3])
            print('----Pooling size ', poolsize[0], ' X ', poolsize[1])
            print('----Input size ',  image_shape[2], ' X ', image_shape[3])
            print('----Input number of feature maps is ',  image_shape[1])
            print('')