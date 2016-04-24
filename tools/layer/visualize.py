import numpy as np
import sys, os
from PIL import Image, ImageFilter

sys.path.append(os.path.abspath("./"))
from storage import ParamStorage

'''
This tool loads the model weight configuration stored in ./results/params.pkl and visualize the weights in the first layer.
The weight configuration of each kernel is converted to a RGB image. The tool assume there are only 64 kernels in the
first layer.
'''

def make_visual(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype(np.uint8)



store = ParamStorage()
data = store.load_params(path="./results/params.pkl")


first_layer = np.array(
    data['params'][-2].eval())
print(first_layer.shape)
#Seems to give weird results, so must read paper it seems like.
i = 0
filters = []
for filter in first_layer:
    filter_image = make_visual(filter)
    #filter_image[0,:,:] = 0
    #filter_image[1,:,:] = 0
    filter_image = np.rollaxis(filter_image, 2)
    filter_image = np.rollaxis(filter_image, 2)
    image = Image.fromarray(filter_image)
    image = image.resize((100, 100), Image.NEAREST)
    #image = image.filter(ImageFilter.GaussianBlur(radius=12))
    filters.append(image)

#64 filters. 100 pixels plus 5 pixel border. 8*8
#or 4*16 = 4 * 100 + 25, 16*100 *
width = 1685
height = 425
filter_showcase = np.zeros((height, width, 3), dtype=np.uint8)
filter_showcase[:, :, :] = (255, 255, 255)
for i in range(5,height, 105):
    for j in range(5, width, 105):

        filter_image = filters.pop()
        pixels = np.array(filter_image.getdata())
        pixels = pixels.reshape(100, 100, 3)
        filter_showcase[i: i+100, j: j+100, :] = pixels[:,:,:]

im = Image.fromarray(filter_showcase)

im.show()