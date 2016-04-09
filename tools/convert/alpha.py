from PIL import Image
import sys, os

sys.path.append(os.path.abspath("./"))

import augmenter.util as loader

#If areas of the dataset is contains artificial white areas. It is beneficial to convert them into RGBA.
#When rotating and sampling , the sample can be checked for alpha values of 0. If one exists the image can be removed.
#This effectively reduce the amount of eligble samples, but will help the model avoid learning useless things like,
#if there are any image borders in the sample.
#
# LABELS NEEDS TO BE MANUALLY COPIED
#
def check(data, x, y):
    for i in range(3):
        if data[x,y][0] < 255:
            return False
    return True

def check_neighborhood(data, x, y, width, height):
    #First center pixel is checked for potentially transparent pixel. Next the neighborhood is check to see if this
    #is not some pixel inside a the actual image which is bright white.
    if check(pixel_data, x, y) is False:
        return False

    empty = 0
    for ny in xrange(y-1, y+1):
        for nx in xrange(x-1, x+1):
            #If outside bounding box, pixel assumed to be transparent.
            if nx < 0 or ny < 0:
                empty += 1
            elif nx > width or ny > height:
                empty += 1
            elif check(data, nx, ny):
                empty += 1
    if empty > 3:
        return True
    return False

def create_dataset_structure(dest, subsets):
    if os.path.exists(dataset_dest):
        raise Exception("Destination folder already exists")
    os.makedirs(dataset_dest)

    for set in subsets:
        sub_dir = dataset_dest + "/" + set
        os.makedirs(sub_dir)
        os.makedirs(sub_dir + "/labels")
        os.makedirs(sub_dir + "/data")


dataset_base = "/home/olav/Pictures/Norwegian_roads_dataset"
dataset_dest = "/home/olav/Pictures/Norwegian_roads_dataset_alpha"
datasets = loader.get_dataset(dataset_base)
color_to_alpha = False
content = ["data"]

create_dataset_structure(dataset_dest, datasets)

for set in datasets:
    for t in content:
        rel_path = "/" + set + "/" + t

        images = loader.get_image_files(dataset_base + rel_path)

        for img_path in images:
            src_path = dataset_base + rel_path + "/" + img_path
            dest_path = dataset_dest + rel_path + "/" + img_path

            src_im = Image.open(src_path)
            im = src_im.convert('RGBA')
            if color_to_alpha:
                pixel_data = im.load()

                height = im.size[1]
                width = im.size[0]
                for y in xrange(height): # For each row ...
                    for x in xrange(width): # Iterate through each column ...

                      if check_neighborhood(pixel_data, x, y, width, height):
                        pixel_data[x, y] = (255, 255, 255, 0)

            im.save(dest_path)