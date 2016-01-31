import time, pickle, gzip
from augmenter import  from_arr_to_data, from_arr_to_label

#Enables dot notation when getting values in config
class Params:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)





def debug_input_data(data, label, data_dim, label_dim, delay=0):
    label_image = from_arr_to_label(label, label_dim)
    data_image= from_arr_to_data(data, data_dim)

    data_image.paste(label_image, (24, 24), label_image)
    data_image = data_image.resize((512, 512))
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




def debug_mnist():
    f = gzip.open('C:\\Users\\olav\\Downloads\\mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    print(train_set[0].shape)
    f.close()