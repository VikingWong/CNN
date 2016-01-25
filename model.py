from abc import ABCMeta, abstractmethod
import theano
import theano.tensor as T
from elements import HiddenLayer, ConvPoolLayer, OutputLayer
import numpy as np
import math
from collections import deque

class AbstractModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, params, verbose):
        #Every layer appended to this variable. layer 0= input, layer N = output
        self.layer= []
        self.L2_layers = []
        self.rng = np.random.RandomState(params.random_seed)
        self.input_data_dim = params.input_data_dim
        self.hidden = params.hidden_layer
        self.output_label_dim = params.output_label_dim

    def get_output_layer(self):
        assert len(self.layer) >0
        return self.layer[-1]

    def get_cost(self, y):
        return  self.get_output_layer().negative_log_likelihood(y)

    def get_errors(self, y):
        self.get_output_layer().errors(y)

    def _weight(self, params, idx):
        if not params:
            return None
        return params[idx]

    def create_predict_function(self, x, data):
        return theano.function([], self.get_output_layer().output,
                   givens={x: data})

    def getL2(self):
        v = 0
        for layer in self.L2_layers:
            v += (layer.W ** 2).sum()
        return v

    @abstractmethod
    def build(self, x, batch_size, init_params=None):
        return



class ShallowModel(AbstractModel):

    def __init__(self, params, verbose=False):
        super(ShallowModel, self).__init__(params, verbose)

    def build(self, x, batch_size, init_params=None):
        print('Shallow neural network model')
        channels, width, height = self.input_data_dim
        layer0 = HiddenLayer(
            self.rng,
            input=x,
            n_in=channels*width*height,
            n_out=2048,
            activation=T.nnet.relu,
            W=self._weight(init_params, 2),
            b=self._weight(init_params, 3)
        )

        layer1 = OutputLayer(
            self.rng,
            input=layer0.output,
            n_in=2048,
            n_out=256,
            W=self._weight(init_params, 0),
            b=self._weight(init_params, 1)
        )

        self.L2_layers = [layer0, layer1]
        self.layer = [layer0, layer1]
        self.params =  layer1.params + layer0.params
        print('Model created!')

#TODO: print number of parameters
class ConvModel(AbstractModel):

    def __init__(self, params, verbose=False):
        super(ConvModel, self).__init__(params, verbose)
        self.nr_kernels = params.nr_kernels
        self.dropout_rate = params.hidden_dropout
        self.conv = params.conv_layers
        #Because of for loop -1 will disappear, but keep queue len being 2.
        self.queue = deque([self.input_data_dim[0], -1])


    def _get_filter(self, next_kernel, filter):
        self.queue.appendleft(next_kernel)
        self.queue.pop()
        return list(self.queue) + list(filter)


    def build(self, x, batch_size, init_params=None):

        print('Convolutional neural network model')
        channels, width, height = self.input_data_dim
        layer_input = x.reshape((batch_size, channels, width, height))

        #See model for explanation
        p_len = 0
        if init_params:
            p_len = len(init_params)

        inp_shape = (batch_size, channels, width, height)

        for i in range(len(self.conv)):
            init_idx = p_len - (i*2)-1

            filter = self._get_filter(self.nr_kernels[i], self.conv[i]["filter"])
            layer = ConvPoolLayer(
                self.rng,
                input=layer_input,
                image_shape=inp_shape,
                filter_shape=filter,
                strides=self.conv[i]["stride"],
                poolsize=self.conv[i]["pool"],
                activation=T.nnet.relu,
                W=self._weight(init_params, init_idx-1),
                b=self._weight(init_params, init_idx)
            )

            layer_input = layer.output
            dim_x = int(math.floor((inp_shape[2] - self.conv[i]["filter"][0] +1) / (self.conv[i]["stride"][0] * self.conv[i]["pool"][0])))
            dim_y = int(math.floor((inp_shape[3] - self.conv[i]["filter"][1] +1) / (self.conv[i]["stride"][1] * self.conv[i]["pool"][1])))

            inp_shape = (batch_size, self.nr_kernels[i], dim_x, dim_y)
            self.layer.append(layer)

        hidden_input = self.layer[-1].output.flatten(2)

        # construct a fully-connected sigmoidal layer
        hidden_layer = HiddenLayer(
            self.rng,
            input=hidden_input,
            n_in=self.nr_kernels[-1] * inp_shape[2] * inp_shape[3],
            n_out=self.hidden,
            activation=T.nnet.relu,
            W=self._weight(init_params, 2),
            b=self._weight(init_params, 3),
            dropout_rate=self.dropout_rate

        )

        output_dim = self.output_label_dim[0] * self.output_label_dim[1]
        output_layer = OutputLayer(
            self.rng,
            input=hidden_layer.output,
            n_in=self.hidden,
            n_out=output_dim,
            W=self._weight(init_params, 0),
            b=self._weight(init_params, 1)
        )

        self.L2_layers = [hidden_layer, output_layer]
        self.layer.extend(self.L2_layers)
        self.params = []
        for layer in reversed(self.layer):
            self.params += layer.params
        print('Model created!')