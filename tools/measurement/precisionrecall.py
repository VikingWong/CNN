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
TODO: All patches instead of random samples per image (optional) in creator
TODO: Implement within x pixel , done by Hinton and Minh.
TODO: Upload to database, and display using rickshaw. (Should be done)
TODO: Do this test at the end of a training session, so PR curves for all experiments automatically (optional)
TODO: Save points to file.
'''

class PrecisionRecallCurve(object):

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
        '''
        Using the params.pkl or instantiated model to create patch predictions.
        '''
        #TODO: What if model is already instantiated?
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
        '''
        Precision and recall found for different threshold values. For each value a binary output image is made.
        The threshold indicate that for a pixel value above threshold value is considered a road pixel.
        This generate different values for precision and recall and highlight the trade off between precision and recall.
        '''
        tests = np.arange(0.1 , 1, 0.05)
        datapoints = []
        for threshold in tests:
            binary_arr = np.ones(predictions.shape)
            low_values_indices = predictions < threshold  # Where values are low
            binary_arr[low_values_indices] = 0  # All low values set to 0

            precision = self._get_precision(labels, binary_arr)
            recall = self._get_recall(labels, binary_arr)
            datapoints.append({"precision": precision, "recall": recall, "threshold": threshold})
        return datapoints


    def _get_precision(self, labels, thresholded_output):
        '''
        Precision between label and output at threshold t.
        Calculate the accuracy of road pixel detection.
        First all positives are counted from output, as well as the true positive. That is road pixels both marked
        in the label and the output. All positives minus true positive gives the false positives. That is predicted
        road pixels which is not marked on the label.
        '''
        #TODO: implement precision with no pixel slack
        total_positive = np.count_nonzero(thresholded_output)
        true_positive = np.count_nonzero(np.array(np.logical_and(labels,  thresholded_output), dtype=np.uint8))
        return true_positive / float(total_positive)


    def _get_recall(self, labels, thresholded_output):
        '''
        Recall between label and output at threshold t.
        See the degree of which the prediction include all positive examples in label.
        So first all postive instances in label are counted (road pixels)
        Then the label and output is compared: In cells where both label and output are one, is
        considered an successful extraction. If output cells are all 1, for all postive pixels in label, the
        recall rate will be 1. If output misses some road pixels this rate will decline.
        '''
        #TODO: implement recall with no pixel slack.
        total_positive = np.count_nonzero(labels)
        true_positive = np.count_nonzero(np.array(np.logical_and(labels,  thresholded_output), dtype=np.uint8))
        return true_positive / float(total_positive)

