import time, pickle, gzip
from augmenter.util import from_arr_to_data, from_arr_to_label
import numpy as np
import Image

#Enables dot notation when getting values in config
class Params:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)





def debug_input_data(data, label, data_dim, label_dim, delay=0):
    label_image = from_arr_to_label(label, label_dim)
    data_image= from_arr_to_data(data, data_dim)

    data_image.paste(label_image, (24, 24), label_image)
    data_image = data_image.resize((128, 128))
    data_image.show()
    time.sleep(delay)

#TODO: TEST THIS, and refactor
def show_debug_sample(data, label, predictions, data_dim, label_dim, std=1):
    img = []
    lbl = []
    for i in range(len(data)):
        prediction_image = from_arr_to_label(predictions[i], label_dim)
        label_image = from_arr_to_label(label[i], label_dim)
        temp = data[i] * std
        min_pixel = np.amin(temp)
        temp = temp + min_pixel

        data_image = from_arr_to_data(temp, data_dim)
        data_image.paste(prediction_image, (24, 24), prediction_image)
        data_image = data_image.resize((128, 128))
        img.append(data_image)

        lbldata_image = from_arr_to_data(temp, data_dim)
        lbldata_image.paste(label_image, (24, 24), label_image)
        lbldata_image = lbldata_image.resize((128, 128))
        lbl.append(lbldata_image)

    new_im = Image.new('RGB', (256,len(img)*128))
    for j in range(len(img)):
        new_im.paste(img[j], (0, j*128))
        new_im.paste(lbl[j], (128, j*128))
    new_im.show()



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

def debug_mnist():
    f = gzip.open('C:\\Users\\olav\\Downloads\\mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    print(train_set[0].shape)
    f.close()