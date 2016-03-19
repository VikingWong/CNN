import numpy as np
import sys, os
from PIL import Image

sys.path.append(os.path.abspath("./"))
from storage import ParamStorage

def make_visual(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype(np.uint8)

#TODO: Visualize first layer. Read paper.
#First undestanding. take snapshots of filters, average input of image to see what the neuron reponds to.
#Second understanding. Take weights, and just visualize the filter.

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")

first_layer = np.array(
    data['params'][-2].eval())
print(first_layer.shape)
#Seems to give weird results, so must read paper it seems like.
i = 0
for filter in first_layer:
    print(filter.shape)
    filter_image = make_visual(filter)
    filter_image = np.rollaxis(filter_image, 2)
    filter_image = np.rollaxis(filter_image, 2)
    image = Image.fromarray(filter_image)
    image = image.resize((256, 256), Image.ANTIALIAS)
    image.show()
    i += 1
    if i > 20:
        break
