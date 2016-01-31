import os
import numpy as np
from PIL import Image
from printing import print_error

def normalize(data, std):
    m = np.mean(data)
    data = (data - m) / std
    return data

def get_image_files(path):
        print('Retrieving {}'.format(path))
        included_extenstions = ['jpg','png', 'tiff', 'tif']
        files = [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]
        files.sort()
        return files


def get_dataset(path):
    content = os.listdir(path)
    if not all(x in ['train', 'valid', 'test'] for x in content):
        print_error('Folder does not contain image or label folder. Path probably not correct')
        raise Exception('Fix dataset_path in config')
    content.sort()
    return content


def from_arr_to_label(label, label_dim):
    label_arr = label.reshape(label_dim, label_dim)
    label_arr = label_arr* 255
    label_arr = [label_arr, label_arr, label_arr, np.ones((16,16))*255]
    label_arr = np.array(label_arr, dtype=np.uint8)
    label_arr = np.rollaxis(label_arr, 2)
    label_arr = np.rollaxis(label_arr, 2)
    return Image.fromarray(label_arr)


def from_arr_to_data(data, data_dim):
    data_arr = data.reshape(3, data_dim, data_dim)
    data_arr = np.rollaxis(data_arr, 2)
    data_arr = np.rollaxis(data_arr, 2)
    data_arr = data_arr * 255
    data_arr = np.array(data_arr, dtype=np.uint8)
    return Image.fromarray(data_arr)


def from_rgb_to_arr(image):
    arr =  np.asarray(image, dtype='float32') / 255
    arr = np.rollaxis(arr, 2, 0)
    arr = arr.reshape(3 * arr.shape[1] * arr.shape[2])
    return arr

def create_image_label(image, dim_data, dim_label):
        #TODO: Euclidiean to dist, ramp up to definite roads. Model label noise in labels?
        y_size = dim_label
        padding = (dim_data - y_size)/2
        #label = np.array(image.getdata())

        #label = np.asarray(image, dtype=theano.config.floatX)
        label = np.asarray(image)
        label = label[padding : padding+y_size, padding : padding+y_size ]
        label = label.reshape(y_size*y_size)

        label = label / 255.0
        return label