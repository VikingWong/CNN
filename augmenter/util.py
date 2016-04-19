import os, random
import numpy as np
from PIL import Image, ImageDraw
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

def create_threshold_image(image, threshold):
    binary_arr = np.ones(image.shape)
    low_values_indices = image <= threshold  # Where values are low
    binary_arr[low_values_indices] = 0  # All low values set to 0
    return binary_arr

def get_sum_road(image):
    arr = np.array(image)
    return np.sum(arr == 255)

def get_road_position(image):
    arr = np.array(image)
    return np.where(arr == 255)

def add_artificial_road_noise(image, threshold):
    label = image.copy()
    nr_road = get_sum_road(label)
    #If there are no road class there is no use in removing some.

    if nr_road == 0:
        return label, 0

    dr = ImageDraw.Draw(label)
    shape_max = int(image.size[0] / 10)
    shape_min = int(image.size[0]/20)
    locations = get_road_position(label)
    location_x = label.size[0]
    location_y = label.size[1]

    removed_threshold = np.clip(threshold, 0.0, 1.0)
    p_roads_removed = 0.0
    while p_roads_removed < removed_threshold:
        i = random.randint(0, locations[0].shape[0] -1 )
        y = locations[0][i]
        x = locations[1][i]
        w = int(random.randint(shape_min, shape_max)/2)
        h = int(random.randint(shape_min, shape_max)/2)
        cor = (x-w, y-h, x+w, y+h)
        dr.ellipse(cor, fill="black")
        nr_artificial_road = get_sum_road(label)
        p_roads_removed = 1.0 - (nr_artificial_road/ float(nr_road))


    return label, p_roads_removed