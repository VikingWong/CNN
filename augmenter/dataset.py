import os, sys
from PIL import Image, ImageFilter

import augmenter.util as util

class Dataset(object):
    '''
    Helper object, that uses os methods to check validity of test, valid or train dataset.
    Collect all image files and base path. Reduce property used to limit the sampling rate.
    '''
    def __init__(self, name, base, folder, reduce):
        self.name = name
        self.base = os.path.join(base, folder)
        self.img_paths = self._get_image_files(self.base)
        self.reduce = reduce
        self.nr_img = len(self.img_paths)


    def open_image(self, i):
        image_path, label_path = self.img_paths[i]
        im = Image.open(os.path.join(self.base, 'data',  image_path), 'r').convert('RGBA')
        la = Image.open(os.path.join(self.base, 'labels',  label_path), 'r').convert('L')
        return im, la


    def _get_image_files(self, path):
        '''
        Each path should contain a data and labels folder containing images.
        Creates a list of tuples containing path name for data and label.
        '''
        tiles = util.get_image_files(os.path.join(path, 'data'))
        labels = util.get_image_files(os.path.join(path, 'labels'))

        self._is_valid_dataset(tiles, labels)
        return list(zip(tiles, labels))


    def _is_valid_dataset(self, tiles, labels):
        if len(tiles) == 0 or len(labels) == 0:
            raise Exception('Data or labels folder does not contain any images')

        if len(tiles) != len(labels):
            raise Exception('Not the same number of tiles and labels')

        for i in range(len(tiles)):
            if os.path.splitext(tiles[i])[0] != os.path.splitext(labels[i])[0]:
                raise Exception('tile', tiles[i], 'does not match label', labels[i])

