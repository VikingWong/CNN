import os, sys
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath("./"))
import augmenter.util as util


'''
Tool to find std of a dataset. This can then be put into the config area.
Need to do this before running cnn because it's expensive to go through all the images.
'''

def get_image_estimate(image):
    image_arr = np.asarray(image)
    image_arr = image_arr.reshape(image_arr.shape[0]*image_arr.shape[1], image_arr.shape[2])
    channels = image_arr.shape[1]
    #Only valid pixels that are not transparent is used to calculate STD
    if channels == 4:
        temp = image_arr[:,3] > 0
        new_arr = image_arr[temp]
        new_arr = new_arr[:, 0:3]
    else:
        new_arr = image_arr

    new_arr = new_arr/255.0
    arr = new_arr.reshape(new_arr.shape[0] * new_arr.shape[1])

    np.random.shuffle(arr)
    return np.array(arr[0:1000]), np.var(new_arr)

def calculate_std_from_dataset(path, dataset):
    #dataset = util.get_dataset(path)

    variance = []
    #for set in dataset:
    folder_path = os.path.join(path, dataset,  'data')
    tiles = util.get_image_files(folder_path)

    samples = []

    i = 0
    for tile in tiles:
        i += 1
        print(tile)
        im = Image.open(os.path.join(folder_path, tile), 'r')
        s, v = get_image_estimate(im)

        samples.append(s)
        variance.append(v)


        if(i%10 == 0):
            print("Progress", i)



    #Average standard deviation by averaging the variances and taking square root of average.
    #Source: http://bit.ly/1VS5pmT


    combined = np.concatenate(samples)
    print("Real std" , np.std(combined))

    dataset_std = np.sqrt(np.sum(variance) / len(variance))

    return dataset_std
path = '/home/olav/Pictures/Norwegian_roads_dataset'
dataset = "train"
if not path or len(path) < 1:
    path = ""
std = calculate_std_from_dataset(path, dataset)
print(std)