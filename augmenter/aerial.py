__author__ = 'Olav'

import numpy as np
import os, math, random
import theano
from PIL import Image, ImageFilter
import augmenter.util as util


class Creator(object):
    '''
    Dynamically load and convert data to appropriate format for theano.
    '''
    def __init__(self, dataset_path, dim=(64, 16), rotation=False, preproccessing=True, only_mixed=False, std=1, mix_ratio=0.5, reduce=1):
        self.dim_data = dim[0]
        self.dim_label = dim[1]
        self.only_mixed_labels = only_mixed #Only use labels containing positive label (roads etc)
        self.rotation = rotation
        self.preprocessing = preproccessing
        self.mix_ratio = mix_ratio
        self.std = std
        self.reduce = reduce
        self.dataset_path = dataset_path
        #Load paths to all images found in dataset


        self.print_verbose()


    def _load_dataset(self):
        test_path, train_path, valid_path = util.get_dataset(self.dataset_path)
        no_reduce = 1
        self.test = Dataset("Test set", self.dataset_path, test_path, no_reduce)
        self.train = Dataset("Training set", self.dataset_path, train_path, reduce)
        self.valid = Dataset("Validation set", self.dataset_path, valid_path, no_reduce)

    def dynamically_create(self, samples_per_image):
        self._load_dataset()

        print('{}# test img, {}# train img, {}# valid img'.format(
            self.test.nr_img, self.train.nr_img, self.valid.nr_img))

        test = self.sample_data(self.test, samples_per_image, mixed_labels=self.only_mixed_labels)
        train = self.sample_data(self.train, samples_per_image,
                                  mixed_labels=self.only_mixed_labels, rotation=self.rotation)
        valid = self.sample_data(self.valid, samples_per_image, mixed_labels=self.only_mixed_labels)

        return train, valid, test


    def sample_data(self, dataset, samples_per_images, mixed_labels=False, rotation=False):
        '''
        Use paths to open data image and corresponding label image. Can apply random rotation, and then
        samples samples_per_images amount of images which is returned in data and label array.
        '''
        #TODO: Support several samplers, IE, random and fully. Change behavior of sampling.
        nr_class = 0
        nr_total = 0

        dropped_images = 0
        idx = 0

        dim_data = self.dim_data
        dim_label = self.dim_label
        max_arr_size = dataset.nr_img *samples_per_images
        data = np.empty((max_arr_size, dim_data*dim_data*3), dtype=theano.config.floatX)
        label = np.empty((max_arr_size, self.dim_label*self.dim_label), dtype=theano.config.floatX)


        print('')
        print('Sampling examples for {}'.format(dataset.base))

        for i in range(dataset.nr_img):
            im, la = dataset.open_image(i)

            width, height = im.size
            width = width - dim_data
            height = height - dim_data

            rot = 0
            if rotation:
                rot = random.uniform(0.0, 360.0)

            image_img = np.asarray(im.rotate(rot))
            label_img = np.asarray(la.rotate(rot))

            s = samples_per_images
            invalid_selection = 0

            #TODO: Check if can get stuck, especially mixed labels.
            while s>0:
                #if invalid_selection > 300:
                #    print("INDVALID SELECTION")
                #    dropped_images += 1
                #    break

                x = random.randint(0, width)
                y = random.randint( 0, height)

                data_temp =     image_img[y : y+dim_data, x : x+dim_data,]
                label_temp =    label_img[y : y+dim_data, x : x+dim_data]

                data_sample =   util.from_rgb_to_arr(data_temp)
                label_sample =  util.create_image_label(label_temp, dim_data, dim_label)

                if self.preprocessing:
                    data_sample = util.normalize(data_sample, self.std)

                #TODO: must shrink numpy.array to reintroduce this
                #if(data_sample.max() == 0 or data_sample.min() == 1):
                #    invalid_selection += 1
                #    continue

                nr_total += 1
                contains_class = not label_sample.max() == 0
                nr_class += int(contains_class)
                if mixed_labels and contains_class and nr_class/float(nr_total) < self.mix_ratio:
                    nr_total -= 1
                    nr_class -= int(contains_class)
                    invalid_selection += 1
                    continue

                data[idx] = data_sample
                label[idx] = label_sample
                idx += 1
                s -= 1

            if i % 50 == 0:
                print('---- Input image: {}/{}'.format(i, dataset.nr_img))

            if im and la:
                del im
                del la

        print("---- Extracted {} images from {}".format(data.shape[0], dataset.name))
        print("---- Images containing class {}/{}".format(nr_class, nr_total))
        print("---- Dropped {} images".format(dropped_images))
        return data, label


    def print_verbose(self):
        print('Initializing dataset creator')
        print('---- Data size {}x{}'.format( self.dim_data, self.dim_data))
        print('---- Label size {}x{}'.format( self.dim_label, self.dim_label))
        print('---- Rotation: {}, preprocessing: {}, and with std: {}'.format(self.rotation, self.preprocessing, self.std))
        if self.only_mixed_labels:
            print('---- CAUTION: will only include labels containing class of interest')
            #print("Image that contains a lot of deadspace in terms of white or dark areas are dropped")
        print('')


class Dataset(object):
    '''
    Helper object, that uses os methods to check validity of a datasets test, valid or train dataset.
    Collect all image files and base path.
    '''
    def __init__(self, name, base, folder, reduce):
        self.name = name
        self.base = os.path.join(base, folder)
        self.img_paths = self._get_image_files(self.base, reduce)
        self.nr_img = len(self.img_paths)


    def open_image(self, i):
        image_path, label_path = self.img_paths[i]
        im = Image.open(os.path.join(self.base, 'data',  image_path), 'r')
        la = Image.open(os.path.join(self.base, 'labels',  label_path), 'r').convert('L')
        return im, la


    def _get_image_files(self, path, reduce_dataset):
        '''
        Each path should contain a data and labels folder containing images.
        Creates a list of tuples containing path name for data and label.
        '''
        tiles = util.get_image_files(os.path.join(path, 'data'))
        labels = util.get_image_files(os.path.join(path, 'labels'))

        self._is_valid_dataset(tiles, labels)

        limit = int(math.floor(reduce_dataset * len(tiles)))
        return list(zip(tiles[0:limit], labels[0:limit]))


    def _is_valid_dataset(self, tiles, labels):
        if len(tiles) == 0 or len(labels) == 0:
            raise Exception('Data or labels folder does not contain any images')

        if len(tiles) != len(labels):
            raise Exception('Not the same number of tiles and labels')

        for i in range(len(tiles)):
            if os.path.splitext(tiles[i])[0] != os.path.splitext(labels[i])[0]:
                raise Exception('tile', tiles[i], 'does not match label', labels[i])

