

import os, sys
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath("./"))
import augmenter.util as util


'''
Tool to find std of a dataset. This can then be put into the config area.
Need to do this before running cnn because it's expensive to go through all the images.
'''


def calculate_std_from_dataset(path):
    dataset = util.get_dataset(path)

    variance = []
    for set in dataset:
        folder_path = os.path.join(path, set,  'data')
        tiles = util.get_image_files(folder_path)
        for tile in tiles:
            print(tile)
            im = Image.open(os.path.join(folder_path, tile), 'r')
            image_arr = np.asarray(im)
            image_arr = image_arr.reshape(image_arr.shape[0]*image_arr.shape[1], image_arr.shape[2])

            #Only valid pixels that are not transparent is used to calculate STD
            temp = image_arr[:,3] > 0
            newarr = image_arr[temp]
            image_arr = image_arr/255
            variance.append(np.var(image_arr))


    #Average standard deviation by averaging the variances and taking square root of average.
    #Source: http://bit.ly/1VS5pmT
    dataset_std = np.sqrt(np.sum(variance) / len(variance))
    return dataset_std
path = '/home/olav/Pictures/Mass_roads_alpha'
if not path or len(path) < 1:
    path = ""
std = calculate_std_from_dataset(path)
print(std)