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
        print("Retrieving", path)
        included_extenstions = ['jpg','png', 'tiff', 'tif']
        return [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]

def from_arr_to_label(label, label_dim):
    label_arr = label.reshape(label_dim, label_dim)
    label_arr = label_arr* 255
    label_arr = np.array(label_arr, dtype=np.uint8)
    return Image.fromarray(label_arr)

def from_arr_to_data(data, data_dim):
    data_arr = data.reshape(3, data_dim, data_dim)
    data_arr = np.rollaxis(data_arr, 2)
    data_arr = np.rollaxis(data_arr, 2)
    data_arr = data_arr * 255
    data_arr = np.array(data_arr, dtype=np.uint8)
    return Image.fromarray(data_arr)

def debug_input_data(data, label, data_dim, label_dim):
    label_image = from_arr_to_label(label, label_dim)
    data_image= from_arr_to_data(data, data_dim)

    label_image.show()
    data_image.show()
    time.sleep(1)




def debug_mnist():
    f = gzip.open('C:\\Users\\olav\\Downloads\\mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    print(train_set[0].shape)
    f.close()

debug_mnist()