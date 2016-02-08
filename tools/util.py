__author__ = 'olav'

import sys, os
import theano.tensor as T
import theano
import numpy as np

sys.path.append(os.path.abspath("./"))
from wrapper import create_output_func
from model import ConvModel


def create_threshold_image(image, threshold):
    binary_arr = np.ones(image.shape)
    low_values_indices = image <= threshold  # Where values are low
    binary_arr[low_values_indices] = 0  # All low values set to 0
    return binary_arr


def create_predictor(dataset, model_config, model_params, batch_size):
    x = T.matrix('x')
    y = T.imatrix('y')
    index = T.lscalar()
    model = ConvModel(model_config, verbose=False)
    model.build(x, batch_size, init_params=model_params)
    return create_output_func(dataset, x, y, [index], model.get_output_layer(), batch_size)


def batch_predict(predictor, dataset, dim, batch_size):
    examples = dataset[0].eval().shape[0]
    nr_of_batches = int(examples/ batch_size)
    result_output = np.empty((examples, dim*dim), dtype=theano.config.floatX)
    result_label = np.empty((examples, dim*dim), dtype=theano.config.floatX)

    for i in range(nr_of_batches):
        output, label = predictor(i)
        result_output[i*batch_size: (i+1)*batch_size] = output
        result_label[i*batch_size: (i+1)*batch_size] = label

    return result_output, result_label
