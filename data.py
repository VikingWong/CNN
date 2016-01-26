from abc import ABCMeta, abstractmethod
import os, gzip, pickle, math,sys
import numpy as np
import theano
import theano.tensor as T

from util import debug_input_data, print_section, print_error
from augmenter.aerial import Creator


class AbstractDataset(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        self.set = {
            'train': None,
            'validation': None,
            'test': None,
        }
        self.all_training = []
        self.active = []

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

    def switch_active_training_set(self, idx):
        '''
        Each epoch a large number of examples will be seen by model. Often all examples will not fit on the GPU at
        the same time. This method, switches the data that are currently reciding in the gpu. Will be called
        nr_of_chunks times per epoch.
        '''
        print('----Changing active chunk')
        new_chunk_x, new_chunk_y = self.all_training[idx]
        self.active[0].set_value(new_chunk_x)
        self.active[1].set_value(new_chunk_y)

    def get_chunk_number(self):
        return len(self.all_training)

    def get_elements(self, idx):
        return len(self.all_training[idx][0])

    def get_total_number_of_batches(self, batch_size):
        s = sum(len(c[0]) for c in self.all_training)
        return math.ceil(s/batch_size)

    def _chunkify(self, dataset, nr_of_chunks, batch_size):

        #Round items per chunk down until there is an exact number of minibatches. Multiple of batch_size
        items_per_chunk = len(dataset[0])/ nr_of_chunks
        if(items_per_chunk < batch_size):
            print_error('Chunk limit in config set to small, or batch size to large. \n'
                        'Each chunk must include at least one batch.')
            raise Exception('Fix chunk_size and batch_size')
        temp = int(items_per_chunk / batch_size)
        items_per_chunk = batch_size * temp
        data, labels = dataset
        #TODO:do floatX operation twice.
        chunks = [[self._floatX(data[x:x+items_per_chunk]), self._floatX(labels[x:x+items_per_chunk])]
                         for x in xrange(0, len(dataset[0]), items_per_chunk)]
        return chunks


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

    def load(self, dataset_path, params, batch_size=1):
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
            train, valid, test = creator.dynamically_create(dataset_path, samples_per_image, reduce=reduce)

        print('')
        print('Preparing shared variables for datasets')
        print('---- Image data shape: {}, label data shape: {}'.format(train[0].shape, train[1].shape))
        print('---- Max chunk size of {}mb'.format(chunks))

        mb = 1000000.0
        train_size = sum(data.nbytes for data in train) / mb
        valid_size = sum(data.nbytes for data in valid) / mb
        test_size = sum(data.nbytes for data in test) / mb
        nr_of_chunks = math.ceil(train_size/chunks)

        print('---- Minimum number of training chunks: {}'.format(nr_of_chunks))
        print('---- Dataset at least:')
        print('---- Training: \t {}mb'.format(train_size))
        print('---- Validation: {}mb'.format(valid_size))
        print('---- Testing: \t {}mb'.format(test_size))

        training_chunks = self._chunkify(train, nr_of_chunks, batch_size)
        print('---- Actual number of training chunks: {}'.format(len(training_chunks)))
        print('---- Elements per chunk: {}'.format(len(training_chunks[0][0])))
        print('---- Last chunk size: {}'.format(len(training_chunks[-1][0])))

        #TODO: Chunkify for validation and testing as well?

        self.active = self._shared_dataset(training_chunks[0], cast_to_int=False)
        self.set['train'] = self.active[0], T.cast(self.active[1], 'int32')
        self.set['validation'] = self._shared_dataset(valid, cast_to_int=True )
        self.set['test'] = self._shared_dataset(test, cast_to_int=True)

        self.all_training = training_chunks #Not stored on the GPU, unlike the shared variables defined above.
        return True



