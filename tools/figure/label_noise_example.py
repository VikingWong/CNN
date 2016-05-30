import sys, os

sys.path.append(os.path.abspath("./"))

import augmenter.util as util
from augmenter.dataset import Dataset
'''
Tool adds artifical noise to an example. Illustrates artifically increasing levels of label noise.
Tool properties, dataset_path, set, image_idx and noise decide the dataset, which set, what image and the amount of noise
to add.
'''

dataset_path = '/home/olav/Pictures/Mass_roads_alpha'
set = 'train'
img_idx = 5
noise = 0.4

d = Dataset("Training set", dataset_path, set, 1.0)

im, la = d.open_image(img_idx)
la, p_removed = util.add_artificial_road_noise(la, noise)
print p_removed
la.save('noise'+str(noise) + '.png')
