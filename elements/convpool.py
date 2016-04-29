import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.sandbox.cuda import dnn
from theano.tensor.signal import pool
import numpy as np
from elements.util import BaseLayer

#TODO: uses deprecated conv and downsample methods. Bleeding edge Theano have convOp and pool2d something.
class ConvPoolLayer(BaseLayer):
    def __init__(self, rng, input, filter_shape, image_shape, drop, poolsize=(2,2), strides=(1, 1),
                 activation=T.tanh, W = None, b = None, verbose = True, dropout_rate=1.0):
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
        super(ConvPoolLayer, self).__init__(rng, input, dropout_rate)
        assert image_shape[1] == filter_shape[1]
        self._verbose_print(verbose, filter_shape, poolsize, image_shape, strides, dropout_rate)

        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
               np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.set_weight(W, -W_bound, W_bound, filter_shape)
        self.set_bias(b, filter_shape[0])
        print(strides)
        if strides[0] == 1 and strides[1] == 1:
            #Strides make the system run impossibly slow because of legacy OP.
            print("No stride, use default conv2d")
            conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            )

        else:
            #When using stride/subsample the system require a GPU and CUDA. Using GPU OP directly.
            #he memory layout to use is bc01, that is batch, channel, first dim, second dim in that order.
            print("DNN convolution and pooling, stride support")
            conv_out = dnn.dnn_conv(input, self.W, subsample=strides)
            #pooled_out = dnn.dnn_pool(conv_out, poolsize, stride=poolsize)

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            st=poolsize,
            ignore_border=True,
            mode='max'
        )

        out = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        droppedOutput = self.dropout(out, dropout_rate)

        self.output = T.switch(T.neq(drop, 0), droppedOutput, out)

        self.params = [self.W, self.b]


    def _verbose_print(self, is_verbose, filter_shape, poolsize, image_shape, strides, dropout_rate):
        if is_verbose:
            print('Convolutional layer with {} kernels'.format(filter_shape[0]))
            print('---- Kernel size \t {}x{}'.format(filter_shape[2], filter_shape[3]))
            print('---- Pooling size \t {}x{}'.format(poolsize[0], poolsize[1]))
            print('---- Input size \t {}x{}'.format(image_shape[2],image_shape[3]))
            print('---- Stride \t \t {}x{}'.format(strides[0],strides[1]))
            print('---- Input number of feature maps is {}'.format(image_shape[1]))
            print('---- Dropout rate is {}'.format(dropout_rate))
            print('')