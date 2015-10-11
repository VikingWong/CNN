__author__ = 'Olav'

import numpy as np
from abc import ABCMeta, abstractmethod
import os
import gzip
import theano
import theano.tensor as T
from PIL import Image
import math
import random

class Creator(object):
    '''
    Dynamically load and convert data to appropriate format for theano.
    '''
    def __init__(self, rotation=False):
        #TODO: Put in params
        self.dim_label = 16
        self.dim_data = 64
        self.rotation = rotation


    def dynamically_create(self, dataset_path, samples_per_image, reduce=1):
        test_path, train_path, valid_path = self._get_dataset(dataset_path)

        base_test = os.path.join(dataset_path, test_path)
        base_train = os.path.join(dataset_path, train_path)
        base_valid = os.path.join(dataset_path, valid_path)

        no_reduce = 1 #Config decide reduce factor of training set. Tet and validation set not affected
        test_img_paths = self._merge_to_examples(base_test, no_reduce)
        train_img_paths = self._merge_to_examples(base_train, reduce)
        valid_img_paths = self._merge_to_examples(base_valid, no_reduce)

        print(len(test_img_paths), '# test img', len(train_img_paths), "# train img", len(valid_img_paths), "# valid img")

        test = self._sample_data(base_test, test_img_paths, samples_per_image)
        train = self._sample_data(base_train, train_img_paths, samples_per_image)
        valid = self._sample_data(base_valid, valid_img_paths, samples_per_image)

        return train, valid, test


    def _get_dataset(self, path):
        content = os.listdir(path)
        if not all(x in ['train', 'valid', 'test'] for x in content):
            raise Exception('Folder does not contain image or label folder. Path probably not correct')
        return content

    def create_image_data(self, image):
        arr =  np.asarray(image, dtype=theano.config.floatX) / 255
        arr = np.rollaxis(arr, 2, 0)
        arr = arr.reshape(3, arr.shape[1]* arr.shape[2])
        return arr

    def create_image_label(self, image):
        #TODO: Euclidiean to dist, ramp up to definite roads. Model label noise in labels?
        '''
         Converts to numpy with new range (0,1).
        Binary image so all values should be either 0 or 1, but edges might have values in between 0 and 255.
        Convert label matrix to integers and invert the matrix. 1: indicate class being present at that place
        0 : The class not present at pixel location.
        '''
        y_size = self.dim_label
        padding = (self.dim_data - y_size)/2
        #label = np.array(image.getdata())
        label = np.asarray(image) / 255
        label = label[padding : padding+y_size, padding : padding+y_size ]
        label = label.reshape(y_size*y_size)

        label = label / 255
        label = np.floor(label)
        label = label.astype(int)
        #label = 1 - label #No need for mass dataset
        return label


    def _get_image_files(self, path):
        print("Retrieving", path)
        included_extenstions = ['jpg','png', 'tiff', 'tif'];
        return [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]


    def _merge_to_examples(self, path, reduce):
        '''
        Each path should contain a data and labels folder containing images.
        Creates a list of tuples containing path name for data and label.
        '''
        tiles = self._get_image_files(os.path.join(path, 'data'))
        labels = self._get_image_files(os.path.join(path, 'labels'))

        if len(tiles) == 0 or len(labels) == 0:
            raise Exception('Data or labels folder does not contain any images')

        if len(tiles) != len(labels):
            raise Exception('Not the same number of tiles and labels')

        for i in range(len(tiles)):
            if os.path.splitext(tiles[i])[0] != os.path.splitext(labels[i])[0]:
                raise Exception('tile', tiles[i], 'does not match label', labels[i])

        limit = math.floor(reduce * len(tiles))
        return list(zip(tiles[0:limit], labels[0:limit]))


    def _sample_data(self, base, paths, samples_per_images):
        '''
        Use paths to open data image and corresponding label image. Can apply random rotation, and then
        samples samples_per_images amount of images which is returned in data and label array.
        '''
        data = []
        label = []
        dim_data = self.dim_data
        use_rotation = self.rotation
        print("")
        print("Sampling examples for", base)
        for i in range(len(paths)):
            d, v = paths[i]
            im = Image.open(os.path.join(base, 'data',  d), 'r')
            la = Image.open(os.path.join(base, 'labels',  v), 'r').convert('L')

            width, height = im.size
            rot = 0
            if use_rotation:
                rot = random.uniform(0.0, 360.0)

            image_img = np.asarray(im.rotate(rot))
            label_img = np.asarray(la.rotate(rot))

            for s in range(samples_per_images):
                x = (width-dim_data)
                y = ( height-dim_data)
                data_temp =     image_img[y : y+dim_data, x : x+dim_data,]
                label_temp =    label_img[y : y+dim_data, x : x+dim_data,]
                data_sample =   self.create_image_data(data_temp)
                label_sample =  self.create_image_label(label_temp)

                data.append(data_sample)
                label.append(label_sample)


            if i % 20 == 0:
                print("Input image: ", i, '/', len(paths))

            im.close()
            la.close()


        data = np.array(data)
        label = np.array(label)
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        return data, label

#creator = Creator()
#import cProfile
#import re
#path = 'C:\\Users\Olav\\Pictures\\Mass_roads'
#cProfile.runctx('creator.dynamically_create(path, 20, rotation=False)', globals(), locals())