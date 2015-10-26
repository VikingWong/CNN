__author__ = 'Olav'

import numpy as np
import os
import theano
from PIL import Image
import math
import random
import util

class Creator(object):
    '''
    Dynamically load and convert data to appropriate format for theano.
    '''
    def __init__(self, dim=(64, 16), rotation=False, preproccessing=True, only_mixed=False, std=1, mix_ratio=0.5):
        self.dim_data = dim[0]
        self.dim_label = dim[1]
        self.only_mixed_labels = only_mixed #Only use labels containing positive label (roads etc)
        self.rotation = rotation
        self.preprocessing = preproccessing
        self.mix_ratio = mix_ratio
        self.std = std

        self.print_verbose()

    def dynamically_create(self, dataset_path, samples_per_image, reduce=1):
        test_path, train_path, valid_path = util.get_dataset(dataset_path)

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
        #input_debugger(train, 64, 16)

        return train, valid, test


    def create_image_label(self, image):
        #TODO: Euclidiean to dist, ramp up to definite roads. Model label noise in labels?
        y_size = self.dim_label
        padding = (self.dim_data - y_size)/2
        #label = np.array(image.getdata())

        #label = np.asarray(image, dtype=theano.config.floatX)
        label = np.asarray(image)
        label = label[padding : padding+y_size, padding : padding+y_size ]
        label = label.reshape(y_size*y_size)

        label = label / 255
        return label


    def _merge_to_examples(self, path, reduce):
        '''
        Each path should contain a data and labels folder containing images.
        Creates a list of tuples containing path name for data and label.
        '''
        tiles = util.get_image_files(os.path.join(path, 'data'))
        labels = util.get_image_files(os.path.join(path, 'labels'))

        self._is_valid_dataset(tiles, labels)

        limit = int(math.floor(reduce * len(tiles)))
        return list(zip(tiles[0:limit], labels[0:limit]))


    def _is_valid_dataset(self, tiles, labels):
        if len(tiles) == 0 or len(labels) == 0:
            raise Exception('Data or labels folder does not contain any images')

        if len(tiles) != len(labels):
            raise Exception('Not the same number of tiles and labels')

        for i in range(len(tiles)):
            if os.path.splitext(tiles[i])[0] != os.path.splitext(labels[i])[0]:
                raise Exception('tile', tiles[i], 'does not match label', labels[i])


    def _sample_data(self, base, paths, samples_per_images):
        '''
        Use paths to open data image and corresponding label image. Can apply random rotation, and then
        samples samples_per_images amount of images which is returned in data and label array.
        '''

        nr_class = 0
        nr_total= 0

        data = []
        label = []
        dim_data = self.dim_data

        print("")
        print("Sampling examples for", base)

        for i in range(len(paths)):
            d, v = paths[i]
            im = Image.open(os.path.join(base, 'data',  d), 'r')
            la = Image.open(os.path.join(base, 'labels',  v), 'r').convert('L')

            width, height = im.size
            width = width - dim_data
            height = height - dim_data

            rot = 0
            if self.rotation:
                rot = random.uniform(0.0, 360.0)

            image_img = np.asarray(im.rotate(rot))
            label_img = np.asarray(la.rotate(rot))
            s = samples_per_images
            while s>0:
                x = random.randint(0, width)
                y = random.randint( 0, height)

                data_temp =     image_img[y : y+dim_data, x : x+dim_data,]
                label_temp =    label_img[y : y+dim_data, x : x+dim_data]

                data_sample =   util.from_rgb_to_arr(data_temp)
                label_sample =  self.create_image_label(label_temp)

                if self.preprocessing:
                    data_sample = util.normalize(data_sample, self.std)

                if self.only_mixed_labels:
                    nr_total += 1
                    contains_class = label_sample.max() == 0
                    nr_class += int(contains_class)
                    if not contains_class and nr_class/float(nr_total) < self.mix_ratio:
                        nr_total -=1
                        continue


                data.append(data_sample)
                label.append(label_sample)

                s -= 1

            if i % 50 == 0:
                print('Input image: ', i, '/', len(paths))

            if im and la:
                del im
                del la

        data = np.array(data)
        label = np.array(label)

        if self.only_mixed_labels:
            print("Images containing class", nr_class, "of" ,nr_total)
        return data, label


    def print_verbose(self):
        print('Initializing dataset creator')
        print('----Data size', self.dim_data, 'x', self.dim_data)
        print('----Label size', self.dim_label, 'x', self.dim_label)
        print('----Rotation:', self.rotation, ' Preprocessing:', self.preprocessing, 'with std:' , self.std)
        if self.only_mixed_labels:
            print('----CAUTION: will only include labels containing class of interest')
        print('')