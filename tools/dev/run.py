from PIL import Image, ImageDraw
import numpy as np
import sys, os, random, time

sys.path.append(os.path.abspath("./"))

is_batch = False
batch_index = 0
if '-batch' in sys.argv:
    is_batch = True
    idx = sys.argv.index('-batch')
    batch_index = sys.argv[idx+1]
    print("Experiment with batch index {}".format(batch_index))