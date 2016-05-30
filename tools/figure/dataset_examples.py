from PIL import Image
import numpy as np
import sys, os, random

sys.path.append(os.path.abspath("./"))

from augmenter.aerial import Creator
import augmenter.util

'''
Creates, displays and saves a grid of patch examples. Illustrates what a typical patch dataset looks like.
dataset_dir sets the dataset. X_grid, y_grid and padding are properties that controls the grid layout. The
dim_data and dim_label is the patch example sizing. Input has a width and height of dim_data and label has a
width and height of dim_label.
'''
dataset_dir = "/home/olav/Pictures/Norwegian_roads_dataset_vbase"
x_grid = 5
y_grid = 4
dim_data = 64
dim_label = 16
padding = 8

def to_rgb(im, w, h):
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
    return ret

l_pad = (dim_data -dim_label)/2
c = Creator(dataset_dir, preproccessing=False, only_mixed=True)
c.load_dataset()
data, labels = c.sample_data(c.train, 10, mixed_labels=True)

shuffled_index = range(len(data))
random.shuffle(shuffled_index)

width = x_grid*2*dim_data + (padding*x_grid)
height = y_grid*dim_data + (padding*y_grid)
patch_showcase = np.zeros((height, width, 3), dtype=np.uint8)
patch_showcase[:, :, :] = (255, 255, 255)

#Puts the label and images in a grid pattern, which include padding inbetween .
for i in range(0,height, dim_data + padding):
    for j in range(0, width, (dim_data*2) +padding):
        idx = shuffled_index.pop()
        d = data[idx]
        l = labels[idx]*255

        data_image = augmenter.util.from_arr_to_data(d, dim_data)
        pixels = np.array(data_image.getdata())
        pixels = pixels.reshape(dim_data, dim_data, 3)
        patch_showcase[i: i+dim_data, j: j+dim_data, :] = pixels[:,:,:]

        label_image = l.reshape(dim_label, dim_label)
        label_pixels = to_rgb(label_image, dim_label, dim_label)
        #Grey background
        patch_showcase[i: i+dim_data, j+ dim_data: j+(2*dim_data), :] = 170
        #Label on top
        patch_showcase[i+l_pad: i+dim_data-l_pad, j+ dim_data + l_pad: j+(2*dim_data)-l_pad, :] = label_pixels[:,:,:]

        idx += 1

im = Image.fromarray(patch_showcase)

im.show()