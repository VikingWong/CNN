__author__ = 'olav'

import theano
import theano.tensor as T
from elements import LogisticRegression, HiddenLayer, ConvPoolLayer
import numpy as np

class Model(object):

    def __init__(self, nkerns):
        #Every layer appended to this variable. layer 0= input, layer N = output
        self.layer = []
        self.rng = np.random.RandomState(23455)
        self.nkerns = nkerns

    def get_output_layer(self):
        assert len(self.layer) >0
        return self.layer[-1]

    def get_cost(self, y):
        return  self.get_output_layer().negative_log_likelihood(y)

    def get_errors(self, y):
        self.get_output_layer().errors(y)

    #TODO: temp
    def build(self, x, batch_size):
        print('... building the model')
        layer0_input = x.reshape((batch_size, 3, 64, 64))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = ConvPoolLayer(
            self.rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 64, 64),
            filter_shape=(self.nkerns[0], 3, 11, 11),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = ConvPoolLayer(
            self.rng,
            input=layer0.output,
            image_shape=(batch_size, self.nkerns[0], 123, 123),
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            self.rng,
            input=layer2_input,
            n_in=self.nkerns[1] * 4 * 4,
            n_out=4096,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
        self.layer = [layer0, layer1, layer2, layer3]
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
