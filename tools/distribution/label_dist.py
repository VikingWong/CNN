import os,sys
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath("./"))

from interface.command import get_command

'''
This tool counts the percentage of road versus non-road pixels in the label set.
'''

is_dataset_path, dataset_path = get_command('-data', default='/home/olav/Pictures/Mass_roads_alpha')
is_set, set_type = get_command('-set', default='train')

def examine_label_dist(path):
    image = Image.open(path, 'r').convert('L')
    arr =  np.array(image)
    w, h = arr.shape
    arr = np.reshape(arr, w*h)
    positive = np.count_nonzero(arr)
    negative = w*h - positive
    ratio_pos = positive/float((w*h))
    ratio_neg = negative/float((w*h))
    print( "%.4f" % ratio_pos, "area contains other stuff", "%.4f" % ratio_neg, " Of area contains roads")
    return ratio_pos

label_path = os.path.join(dataset_path, set_type, 'labels')
included_extenstions = ['jpg','png', 'tiff', 'tif']
labels = [fn for fn in os.listdir(label_path) if any([fn.endswith(ext) for ext in included_extenstions])]

labels = labels
count = 0
count2 = 0
for label in labels:
    ratio = examine_label_dist(os.path.join(label_path,label))
    count += ratio
print("Percentage of {} label pixel which contain roads: {}".format(set_type, count/len(labels)))