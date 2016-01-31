__author__ = 'olav'
from PIL import Image
import numpy as np
import math, sys, os
import theano.tensor as T
import theano

sys.path.append(os.path.abspath("./"))


'''
TODO: Create all possible patches in test dataset.
TODO: Create predictions using system for all these patches.
TODO: Vary threshold and collect precision and recall for each patch.
TODO: Implement within x pixel , done by Hinton and Minh.
TODO: Upload to database, and display using rickshaw. (Should be done)
TODO: Do this test at the end of a training session, so PR curves for all experiments automatically (optional)
TODO: Save points to file.
'''

class PrecisionRecallCurve(object):
    LABEL_SIZE = 16
    IMAGE_SIZE = 64

    def __init__(self, model, params, std=1):
        self.model = model
        self.params = params
        self.std = std

    def get_curves_datapoints(self, dataset=None):
        if not dataset:
            dataset = self._create_dataset()
            
        predictions = self._predict_patches(dataset)
        datapoints = self._get_datapoints(dataset, predictions)
        return datapoints

    def _create_dataset(self):
        return None #Dataset

    def _predict_patches(self, dataset):
        number = dataset.shape[0] #Might be to big for a single prediction, lol mao.
        x = T.matrix('x')
        shared_x = theano.shared(np.asarray(dataset, dtype=theano.config.floatX), borrow=True)
        self.model.build(x, number, init_params=self.params)
        return None #Predictions for each example

    def _get_datapoints(self, dataset, predictions):
        return None #Datapoints


