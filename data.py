from abc import ABCMeta, abstractmethod
import os
import gzip
import pickle

import numpy as np
import theano
import theano.tensor as T
from util import debug_input_data

from augmenter.aerial import Creator


class AbstractDataset(metaclass=ABCMeta):

    def __init__(self):
        self.set = {
            'train': None,
            'validation': None,
            'test': None,
        }

    @abstractmethod
    def load(self, dataset_path):
        """Loading and transforming logic for dataset"""
        return

    def _floatX(self, d):
        #Creates a data representation suitable for GPU
        return np.asarray(d, dtype=theano.config.floatX)

    def _get_file_path(self, dataset):
        data_dir, data_file = os.path.split(dataset)
        #TODO: Add some robustness, like checking if file is folder and correct that
        assert os.path.isfile(dataset)
        return dataset

    def _shared_dataset(self, data_xy, borrow=True, cast_to_int=True):
        #Stored in theano shared variable to allow Theano to copy it into GPU memory
        data_x, data_y = data_xy
        shared_x = theano.shared(self._floatX(data_x), borrow=borrow)
        shared_y = theano.shared(self._floatX(data_y), borrow=borrow)
        #Since labels are index integers they have to be treated as such during computations.
        #Shared_y is therefore cast to int.
        if cast_to_int:
            print("Casted to int")
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y



class MnistDataset(AbstractDataset):

    def load(self, dataset):
        dataset = self._get_file_path(dataset)
        print("... loading data")
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
        f.close()

        #All the shared variables in a simple datastructure for easier access.
        self.set['test'] = self._shared_dataset(test_set, cast_to_int=True)
        self.set['validation'] = self._shared_dataset(valid_set, cast_to_int=True)
        self.set['train'] = self._shared_dataset(train_set, cast_to_int=True)



        return True #TODO: Implement boolean for whether everything went ok or not


class AerialDataset(AbstractDataset):

    def load(self, dataset_path,params):
        samples_per_image = params.samples_per_image
        use_rotation = params.use_rotation
        reduce = params.reduce
        dim = (params.input_dim, params.output_dim)
        std = params.dataset_std

        #TODO: Handle premade datasets. Later on when dataset structure is finalized
        #TODO: Use shared_value.set_value(my_dataset[...]) when dataset is to big to fit on gpu
        creator = Creator(dim=dim, rotation=use_rotation, preproccessing=True, std=std)
        #get image and label folder from dataset, if valid
        if dataset_path.endswith('.pkl'):
            raise NotImplementedError('Not tested yet')
            f = open(dataset_path, 'rb')
            train, valid, test = pickle.load(f , encoding='latin1')
            f.close()
        else:
            train,valid,test = creator.dynamically_create(dataset_path, samples_per_image, reduce=reduce)

        print('')
        print('Image data shape: ', train[0].shape, 'Label data shape', train[1].shape)
        print('')
        print('Creating shared dataset for train, valid and test')
        self.set['train'] = self._shared_dataset(train)
        self.set['validation'] = self._shared_dataset(valid)
        self.set['test'] = self._shared_dataset(test)

        return True



