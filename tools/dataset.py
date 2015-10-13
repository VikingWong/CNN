__author__ = 'olav'
from PIL import Image
import numpy as np

def examine_label_dist(path):
    image = Image.open(path, 'r').convert('L')
    arr =  np.array(image)
    w, h = arr.shape
    image.close()
    arr = np.reshape(arr, w*h)
    positive = np.count_nonzero(arr)
    negative = w*h - positive
    percent_pos = positive/(w*h)*100
    percent_neg = negative/(w*h)*100
    print("%.2f" % percent_pos, " Of area contains roads")
    print( "%.2f" % percent_neg, "area contains other stuff")

path = 'C:\\Users\\olav\\Pictures\\Mass_roads_overfitting_test\\test\\labels\\17578915_15.tif'
examine_label_dist(path)