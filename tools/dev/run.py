from PIL import Image, ImageDraw
import numpy as np
import sys, os, random

sys.path.append(os.path.abspath("./"))
#from augmenter.util import add_artificial_road_noise

def get_sum_road(image):
    arr = np.array(image)
    return np.sum(arr == 255)

def get_road_position(image):
    arr = np.array(image)
    return np.where(arr == 255)
def add_artificial_road_noise(image, threshold):
    #TODO: get all locations of road pixels, random selection from these for x, y. Better erasing
    label = image.copy()
    nr_road = get_sum_road(label)
    #If there are no road class there is no use in removing some.
    if nr_road == 0:
        return label

    dr = ImageDraw.Draw(label)
    shape_max = int(image.size[0] / 10)
    shape_min = int(image.size[0]/20)
    locations = get_road_position(label)
    location_x = label.size[0]
    location_y = label.size[1]

    removed_threshold = np.clip(threshold, 0.0, 1.0)
    print(removed_threshold)
    p_roads_removed = 0.0
    while p_roads_removed < removed_threshold:
        i = random.randint(0, locations[0].shape[0])
        y = locations[0][i]
        x = locations[1][i]
        w = int(random.randint(shape_min, shape_max)/2)
        h = int(random.randint(shape_min, shape_max)/2)
        cor = (x-w, y-h, x+w, y+h)
        dr.ellipse(cor, fill="black")
        nr_artificial_road = get_sum_road(label)
        p_roads_removed = 1.0 - (nr_artificial_road/ float(nr_road))


    return label, p_roads_removed

image_path = '/home/olav/Pictures/Mass_roads/test/labels/20878930_15.tif'

image = Image.open(image_path, 'r').convert('L')
arr =  np.array(image)
noise_image, p_roads_removed = add_artificial_road_noise(image, 0.1)
print(p_roads_removed)
noise_image.show()
#image.show()