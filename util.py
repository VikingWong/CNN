import os
import numpy as np
from PIL import Image

class Params:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)

def get_image_files(path):
        print("Retrieving", path)
        included_extenstions = ['jpg','png', 'tiff', 'tif']
        return [fn for fn in os.listdir(path) if any([fn.endswith(ext) for ext in included_extenstions])]

def debug_input_data(data, label):
    label_arr = label.reshape(16, 16)
    label_arr = label_arr* 255
    label_arr = np.array(label_arr, dtype=np.uint8)

     data_arr = np.rollaxis(data, 2, 0)
     data_arr = data_arr.reshape(3, data_arr.shape[1]* data_arr.shape[2])