__author__ = 'olav'

import numpy as np
import math, sys, os
import theano.tensor as T
import theano

sys.path.append(os.path.abspath("./"))

from augmenter import Creator
from data import _

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

    def get_curves_datapoints(self, dataset=None):
        if not dataset:
            dataset = self._create_dataset()
        print(dataset)
        raise
        predictions = self._predict_patches(dataset)
        datapoints = self._get_datapoints(dataset, predictions)
        return datapoints

    def _create_dataset(self):
        dim = (self.dataset_config.input_dim, self.dataset_config.output_dim)
        path = self.dataset_path
        preprocessing = self.dataset_config.use_preprocessing
        std = self.dataset_config.dataset_std

        creator = Creator(path, dim=dim, preproccessing=preprocessing, std=std)
        creator.load_dataset()
        test_dataset = creator.sample_data(creator.test, 10)
        test_dataset = self._shared_dataset(test, cast_to_int=True)
        return test_dataset

    def _predict_patches(self, dataset):
        number = dataset.shape[0] #Might be to big for a single prediction, lol mao.
        x = T.matrix('x')
        shared_x = theano.shared(np.asarray(dataset, dtype=theano.config.floatX), borrow=True)
        self.model.build(x, number, init_params=self.params)
        return None #Predictions for each example

    def _get_datapoints(self, dataset, predictions):
        return None #Datapoints


