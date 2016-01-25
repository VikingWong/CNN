from abc import ABCMeta, abstractmethod
import os, gzip, pickle, math
import numpy as np
import theano
import theano.tensor as T

from util import debug_input_data, print_section
from augmenter.aerial import Creator


class AbstractDataset(object):
    __metaclass__ = ABCMeta
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
            print("---- Casted to int")
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y



class MnistDataset(AbstractDataset):

    def load(self, dataset):
        print("Creating MNIST dataset")
        dataset = self._get_file_path(dataset)
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
        f.close()

        #All the shared variables in a simple datastructure for easier access.
        self.set['test'] = self._shared_dataset(test_set, cast_to_int=True)
        self.set['validation'] = self._shared_dataset(valid_set, cast_to_int=True)
        self.set['train'] = self._shared_dataset(train_set, cast_to_int=True)



        return True #TODO: Implement boolean for whether everything went ok or not


class AerialDataset(AbstractDataset):

    def load(self, dataset_path, params):
        print_section('Creating aerial image dataset')
        samples_per_image = params.samples_per_image
        preprocessing = params.use_preprocessing
        use_rotation = params.use_rotation
        reduce = params.reduce
        dim = (params.input_dim, params.output_dim)
        std = params.dataset_std
        mixed = params.only_mixed_labels
        chunks = params.chunk_size

        #TODO: Handle premade datasets. Later on when dataset structure is finalized
        creator = Creator(dim=dim, rotation=use_rotation, preproccessing=preprocessing, std=std, only_mixed=mixed)
        #get image and label folder from dataset, if valid
        if dataset_path.endswith('.pkl'):
            raise NotImplementedError('Not tested yet')
            f = open(dataset_path, 'rb')
            train, valid, test = pickle.load(f , encoding='latin1')
            f.close()
        else:
            train,valid,test = creator.dynamically_create(dataset_path, samples_per_image, reduce=reduce)

        print('')
        print('Preparing shared variables for datasets')
        print('---- Image data shape: {}, label data shape: {}'.format(train[0].shape, train[1].shape))

        mb = 1000000
        train_size = sum(data.nbytes for data in train) / mb
        valid_size = sum(data.nbytes for data in valid) / mb
        test_size = sum(data.nbytes for data in test) / mb

        print('---- Dataset at least:')
        print('---- Training: \t {}mb'.format(train_size))
        print('---- Validation: {}mb'.format(valid_size))
        print('---- Testing: \t {}mb'.format(test_size))

        nr_chunks = math.ceil(train_size/chunks) + 1

        #TODO: use size, to divide list into chunks.
        #TODO: Array of arrays [[],[],] maybe?
        #TODO: Only training set for the moment being.

        print('===========', nr_chunks)
        self.set['train'] = self._shared_dataset(train)
        self.set['validation'] = self._shared_dataset(valid)
        self.set['test'] = self._shared_dataset(test)
        return True



