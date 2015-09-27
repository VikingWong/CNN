import numpy as np
import os
import gzip
import theano
import theano.tensor as T

#TODO: Probably make it a tad bit more general. IE loading datasets not a stored in same format as mnist, and
#formatting the dataset propertly
class Dataset(object):

    def __init__(self):
        self.set = {
            'train': None,
            'validation': None,
            'test': None,
        }

    def load(self, dataset):
        dataset = self._get_file_path(dataset)
        print("... loading data")
        f = gzip.open(dataset, 'rb')
        import pickle
        train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
        f.close()

        #All the shared variables in a simple datastructure for easier access.
        self.set['test'] = self._shared_dataset(train_set)
        self.set['validation'] = self._shared_dataset(train_set)
        self.set['train'] = self._shared_dataset(train_set)
        print(self.set)

        return True #TODO: Implement boolean for whether everything went ok or not

    def _shared_dataset(self, data_xy, borrow=True):
        #Stored in theano shared variable to allow Theano to copy it into GPU memory
        data_x, data_y = data_xy
        shared_x = theano.shared(self._floatX(data_x), borrow=borrow)
        shared_y = theano.shared(self._floatX(data_y), borrow=borrow)
        #Since labels are index integers they have to be treated as such during computations.
        #Shared_y is therefore cast to int.
        return shared_x, T.cast(shared_y, 'int32')

    def _floatX(self, d):
        #Creates a data representation suitable for GPU
        return np.asarray(d, dtype=theano.config.floatX)

    def _get_file_path(self, dataset):
        data_dir, data_file = os.path.split(dataset)
        #TODO: Add some robustness, like checking if file is folder and correct that
        assert os.path.isfile(dataset)
        return dataset