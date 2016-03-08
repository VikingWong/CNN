from abc import ABCMeta, abstractmethod
import os, gzip, pickle, math,sys
import numpy as np
import theano
import theano.tensor as T

from printing import print_section, print_error
from augmenter import Creator


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

        self.nr_examples = {}


    @abstractmethod
    def load(self, dataset_path):
        """Loading and transforming logic for dataset"""
        return

    def destroy(self):
        #Remove contents from GPU
        #TODO: symbolic cast operation, makes set_value not possible.
        for key in self.set:
            self.set[key][0].set_value([[]])
            #self.set[key][1].set_value([[]])
        del self.all_training
        del self.active

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
        chunks = [[AbstractDataset._floatX(data[x:x+items_per_chunk]), AbstractDataset._floatX(labels[x:x+items_per_chunk])]
                         for x in xrange(0, len(dataset[0]), items_per_chunk)]

        #If the last chunk is less than batch size, it is cut. No reason for an unnecessary swap.
        last_chunk_size = len(chunks[-1][0])
        #TODO: Quick fix
        if(last_chunk_size < batch_size*15):
            chunks.pop(-1)
            print('---- Removed last chunk. '
                  '{} elements not enough for at least one minibatch of {}'.format(last_chunk_size, batch_size))
        return chunks


    def get_report(self):
        return self.nr_examples


    def switch_active_training_set(self, idx):
        '''
        Each epoch a large number of examples will be seen by model. Often all examples will not fit on the GPU at
        the same time. This method, switches the data that are currently reciding in the gpu. Will be called
        nr_of_chunks times per epoch.
        '''
        #print('---- Changing active chunk') #This works very well so no need to print it all the time
        new_chunk_x, new_chunk_y = self.all_training[idx]
        self.active[0].set_value(new_chunk_x)
        self.active[1].set_value(new_chunk_y)


    @staticmethod
    def _floatX(d):
        #Creates a data representation suitable for GPU
        return np.asarray(d, dtype=theano.config.floatX)


    @staticmethod
    def _get_file_path(dataset):
        data_dir, data_file = os.path.split(dataset)
        #TODO: Add some robustness, like checking if file is folder and correct that
        assert os.path.isfile(dataset)
        return dataset


    @staticmethod
    def shared_dataset(data_xy, borrow=True, cast_to_int=True):
        #Stored in theano shared variable to allow Theano to copy it into GPU memory
        data_x, data_y = data_xy
        print(data_x.shape)
        print(data_y.shape)
        shared_x = theano.shared(AbstractDataset._floatX(data_x), borrow=borrow)
        shared_y = theano.shared(AbstractDataset._floatX(data_y), borrow=borrow)

        if cast_to_int:
            print("---- Casted to int")
            #Since labels are index integers they have to be treated as such during computations.
            #Shared_y is therefore cast to int.
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y


    @staticmethod
    def _dataset_check(name, dataset, batch_size):
        #If there are are to few examples for at least one batch, the dataset is invalid.
        if len(dataset[0]) < batch_size:
            print_error('Insufficent examples in {}. '
                        '{} examples not enough for at least one minibatch'.format(name, len(dataset[0])))
            raise Exception('Decrease batch_size or increase samples_per_image')


class AerialDataset(AbstractDataset):

    def load(self, dataset_path, params, batch_size=1):
        print_section('Creating aerial image dataset')
        samples_per_image = params.samples_per_image
        preprocessing = params.use_preprocessing
        use_rotation = params.use_rotation
        reduce_training = params.reduce_training
        reduce_testing = params.reduce_testing
        reduce_validation = params.reduce_validation
        dim = (params.input_dim, params.output_dim)
        self.std = params.dataset_std
        mixed = params.only_mixed_labels
        chunks = params.chunk_size

        #TODO: Handle premade datasets. Later on when dataset structure is finalized
        #get image and label folder from dataset, if valid
        if dataset_path.endswith('.pkl'):
            raise NotImplementedError('Not tested yet')
            f = open(dataset_path, 'rb')
            train, valid, test = pickle.load(f, encoding='latin1')
            f.close()
        else:
            creator = Creator(dataset_path,
                              dim=dim,
                              rotation=use_rotation,
                              preproccessing=preprocessing,
                              std=self.std,
                              only_mixed=mixed,
                              reduce_testing=reduce_testing,
                              reduce_training=reduce_training,
                              reduce_validation=reduce_validation)
            train, valid, test = creator.dynamically_create(samples_per_image)

        #Testing dataset size requirements
        AerialDataset._dataset_check('train', train, batch_size)
        AerialDataset._dataset_check('valid', valid, batch_size)
        AerialDataset._dataset_check('test', test, batch_size)

        print('')
        print('Preparing shared variables for datasets')
        print('---- Image data shape: {}, label data shape: {}'.format(train[0].shape, train[1].shape))
        print('---- Max chunk size of {}mb'.format(chunks))

        self.nr_examples['train'] = train[0].shape[0]
        self.nr_examples['valid'] = valid[0].shape[0]
        self.nr_examples['test'] = test[0].shape[0]

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
        training_chunks = self._chunkify(train, nr_of_chunks, batch_size)
        print('---- Actual number of training chunks: {}'.format(len(training_chunks)))
        print('---- Elements per chunk: {}'.format(len(training_chunks[0][0])))
        print('---- Last chunk size: {}'.format(len(training_chunks[-1][0])))

        #TODO: Chunkify for validation and testing as well?
        self.active = AerialDataset.shared_dataset(training_chunks[0], cast_to_int=False)
        self.set['train'] = self.active[0], T.cast(self.active[1], 'int32')
        self.set['validation'] = AerialDataset.shared_dataset(valid, cast_to_int=True )
        self.set['test'] = AerialDataset.shared_dataset(test, cast_to_int=True)

        self.all_training = training_chunks #Not stored on the GPU, unlike the shared variables defined above.
        return True



