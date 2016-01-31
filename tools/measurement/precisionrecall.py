__author__ = 'olav'

import numpy as np
import math, sys, os
import theano.tensor as T
import theano

sys.path.append(os.path.abspath("./"))

from augmenter import Creator
from data import AerialDataset
from wrapper import create_output_func

'''
TODO: Create all possible patches in test dataset.
TODO: Create predictions using system for all these patches.
TODO: Vary threshold and collect precision and recall for each patch.
TODO: Implement within x pixel , done by Hinton and Minh.
TODO: Upload to database, and display using rickshaw. (Should be done)
TODO: Do this test at the end of a training session, so PR curves for all experiments automatically (optional)
TODO: Save points to file.
'''
#TODO: use creator with random images. Maybe implement new sampler in creator.
class PrecisionRecallCurve(object):
    LABEL_SIZE = 16
    IMAGE_SIZE = 64

    def __init__(self, dataset_path, model, model_params, model_config, dataset_config):
        self.model = model
        self.params = model_params
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.dataset_path = dataset_path


    def get_curves_datapoints(self, batch_size, dataset=None):
        if not dataset:
            dataset = self._create_dataset()

        predictions, labels = self._predict_patches(dataset, batch_size)
        datapoints = self._get_datapoints(predictions, labels)
        return datapoints


    def _create_dataset(self):
        dim = (self.dataset_config.input_dim, self.dataset_config.output_dim)
        path = self.dataset_path
        preprocessing = self.dataset_config.use_preprocessing
        std = self.dataset_config.dataset_std
        samples_per_image = 10
        creator = Creator(path, dim=dim, preproccessing=preprocessing, std=std)
        creator.load_dataset()
        #Creating a shared variable of sampled test data
        return AerialDataset.shared_dataset(creator.sample_data(creator.test, samples_per_image), cast_to_int=True)


    def _predict_patches(self, dataset, batch_size):
        x = T.matrix('x')
        y = T.imatrix('y')
        index = T.lscalar()
        self.model.build(x, batch_size, init_params=self.params)
        compute_output = create_output_func(dataset, x, y, [index], self.model.get_output_layer(), batch_size)

        examples = dataset[0].eval().shape[0]
        nr_of_batches = int(examples/ batch_size)
        dim = self.dataset_config.output_dim
        result_output = np.empty((examples*nr_of_batches, dim*dim), dtype=theano.config.floatX)
        result_label = np.empty((examples*nr_of_batches, dim*dim), dtype=theano.config.floatX)

        for i in range(nr_of_batches):
            output, label = compute_output(i)
            result_output[i*batch_size: (i+1)*batch_size] = output
            result_label[i*batch_size: (i+1)*batch_size] = label

        return result_output, result_label

    def _get_datapoints(self, predictions, labels):
        return None #Datapoints


