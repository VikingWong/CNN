import os
import numpy as np
from PIL import Image
import time
import pickle, gzip

class Params:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)


def get_image_files(path):
        print('Retrieving {}'.format(path))
        included_extenstions = ['jpg','png', 'tiff', 'tif']
        files = [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]
        files.sort()
        return files


def get_dataset(path):
    content = os.listdir(path)
    if not all(x in ['train', 'valid', 'test'] for x in content):
        raise Exception('Folder does not contain image or label folder. Path probably not correct')
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
    arr = arr.reshape(3*  arr.shape[1] * arr.shape[2])
    return arr


def debug_input_data(data, label, data_dim, label_dim, delay=0):
    label_image = from_arr_to_label(label, label_dim)
    data_image= from_arr_to_data(data, data_dim)

    data_image.paste(label_image, (24, 24), label_image)
    data_image = data_image.resize( (512,512) )
    data_image.show()
    time.sleep(delay)


def input_debugger(data, data_dim, label_dim):
    n = 0
    length = data[0].shape[0]
    for i in range(length):
        d = data[0][i]
        l = data[1][i]

        m = l.max(0).max(0)
        if(m > 0):
            n = n + 1
            debug_input_data(d, l, data_dim, label_dim)
    print('Total:' , data[0].shape[0], ' over', n)


def normalize(data, std):
    m = np.mean(data)
    data = (data - m) / std
    return data

# ======================PRINTING UTILITIES============================
def debug_mnist():
    f = gzip.open('C:\\Users\\olav\\Downloads\\mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    print(train_set[0].shape)
    f.close()


def print_section(description):
    print('')
    print('')
    print('========== ' + str(description) + ' ==========')


def print_test(epoch, idx, minibatches, loss):
    #print('Epoch {}, minibatch {}/{}'.format(epoch, idx, minibatches))
    print('---- Test error %f MSE' % (loss))
    print('')


def print_valid(epoch, idx, minibatches, loss):
    print('')
    print('Epoch {}, minibatch {}/{}'.format(epoch, idx, minibatches))
    print('---- Validation error %f MSE' % (loss))