__author__ = 'Olav'

import numpy as np
from abc import ABCMeta, abstractmethod
import os
import gzip
import theano
import theano.tensor as T
from PIL import Image
import math

class Creator(object):
    '''
    Dynamically load and convert data to appropriate format for theano.
    '''
    def _get_dataset(self, path):
        content = os.listdir(path)
        if 'data' not in content or 'label' not in content:
            raise Exception('Folder does not contain image or label folder. Path probably not correct')
        return content

    def _create_xy(self, d, l, start, end):
        '''Return dataset in the form data, label'''
        return (d[start:end], l[start:end])

    def create_image_data(self, path):
        image = Image.open(path, 'r')
        arr =  np.asarray(image, dtype=theano.config.floatX) / 255
        arr = np.rollaxis(arr, 2, 0)
        arr = arr.reshape(3, arr.shape[1]* arr.shape[2])
        image.close()
        return arr

    def create_image_label(self, path):
        image = Image.open(path, 'r').convert('L')
        label = np.array(image.getdata())
        #label = np.invert(label)
        #label = np.divide(label, 255 , dtype=theano.config.floatX)
        image.close()
        return label

    def _get_image_files(self, path):
        print("Retrieving", path)
        included_extenstions = ['jpg','png'];
        return [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]


    def dynamically_create(self, dataset_path, percentage, examples_dist):
        folder = self._get_dataset(dataset_path)
        tile_path = os.path.join(dataset_path, folder[0])
        label_path = os.path.join(dataset_path, folder[1])

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

        data = np.array(data)
        label = np.array(label)
        ddim = data.shape
        data = data.reshape(ddim[0], ddim[1]*ddim[2])
        print(data.shape)
        print(label.shape)
        nr_train = int(nr_examples*examples_dist[0])
        nr_valid = int(nr_examples*examples_dist[1])

        train = self._create_xy(data, label, 0, nr_train)
        valid = self._create_xy(data, label, nr_train,nr_train + nr_valid)
        test  = self._create_xy(data, label, nr_train + nr_valid, nr_examples)
        return (train, valid, test)