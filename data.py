import numpy as np
from abc import ABCMeta, abstractmethod
import os
import gzip
import theano
import theano.tensor as T
from PIL import Image
import math

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

    def _shared_dataset(self, data_xy, borrow=True, cast_to_int=False):
        #Stored in theano shared variable to allow Theano to copy it into GPU memory
        data_x, data_y = data_xy
        shared_x = theano.shared(self._floatX(data_x), borrow=borrow)
        shared_y = theano.shared(self._floatX(data_y), borrow=borrow)
        #Since labels are index integers they have to be treated as such during computations.
        #Shared_y is therefore cast to int.
        if cast_to_int:
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y



class MnistDataset(AbstractDataset):

    def load(self, dataset):
        dataset = self._get_file_path(dataset)
        print("... loading data")
        f = gzip.open(dataset, 'rb')
        import pickle
        train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
        f.close()

        #All the shared variables in a simple datastructure for easier access.
        self.set['test'] = self._shared_dataset(test_set, cast_to_int=True)
        self.set['validation'] = self._shared_dataset(valid_set, cast_to_int=True)
        self.set['train'] = self._shared_dataset(train_set, cast_to_int=True)
        print(self.set)

        return True #TODO: Implement boolean for whether everything went ok or not


class AerialDataset(AbstractDataset):

    def load(self, dataset_path, percentage=0.1, examples_dist=(0.8, 0.1, 0.1)):
        '''
        Percentage to specify how much of the dataset to use. 10 percent for good performance when dev
        '''
        #TODO: Use context, smaller output than input. Need to see how this fits in first.
        #TODO: Handle premade datasets. Later on when dataset structure is finalized
        #TODO: Use shared_value.set_value(my_dataset[...]) when dataset is to big to fit on gpu
        #get image and label folder from dataset, if valid
        folder = self._get_dataset(dataset_path)
        tile_path = os.path.join(dataset_path, folder[0])
        label_path = os.path.join(dataset_path, folder[1])
        train,valid,test = self._augment_dataset(tile_path,label_path, percentage, examples_dist)

        self.set['train'] = self._shared_dataset(train)
        self.set['validation'] = self._shared_dataset(valid)
        self.set['test'] = self._shared_dataset(test)
        print(self.set)
        return True

    def _create_xy(self, d, l, start, end):
        '''Return dataset in the form data, label'''
        return (d[start:end], l[start:end])

    def create_image_data(self, path):
        image = Image.open(path, 'r')
        arr =  np.asarray(image, dtype=theano.config.floatX) / 255
        arr = np.rollaxis(arr, 2, 0)
        print(arr)
        raise Exception('reshape')
        image.close()
        return arr

    def create_image_label(self, path):
        image = Image.open(path, 'r')
        label = np.invert(np.asarray(image))
        label = np.divide(label, 255 , dtype=theano.config.floatX)
        image.close()
        return label

    def _get_image_files(self, path):
        print("Retrieving", path)
        included_extenstions = ['jpg','png'];
        return [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]

    def _get_dataset(self, path):
        content = os.listdir(path)
        if 'data' not in content or 'label' not in content:
            raise Exception('Folder does not contain image or label folder. Path probably not correct')
        return content

    def _augment_dataset(self, tile_path,label_path, percentage, examples_dist):
        #TODO: Better name, Actually dynamically loads data etc
        tiles = self._get_image_files(tile_path)
        vectors = self._get_image_files(label_path)

        data = []
        label = []

        nr_examples = math.floor(len(tiles) * percentage)
        print('Input folder contains', len(tiles), 'examples, and only',
              percentage*100, '% is used')
        print('Dataset consists of a total of', nr_examples, 'examples')

        for i in range(nr_examples):
            vector = self.create_image_label(os.path.join(label_path, vectors[i]))
            image = self.create_image_data(os.path.join(tile_path, tiles[i]))
            data.append(image)
            label.append(vector)

            if i % 200 == 0:
                print("Tile: ", i, '/', nr_examples)

        nr_train = int(nr_examples*examples_dist[0])
        nr_valid = int(nr_examples*examples_dist[1])

        train = self._create_xy(data, label, 0, nr_train)
        valid = self._create_xy(data, label, nr_train,nr_train + nr_valid)
        test  = self._create_xy(data, label, nr_train + nr_valid, nr_examples)
        return (train, valid, test)

