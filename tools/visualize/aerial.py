from PIL import Image
import theano.tensor as T
import numpy as np
import theano
import math, sys, os

sys.path.append(os.path.abspath("./"))

import tools.util as util
from data import AerialDataset

from augmenter.util import from_rgb_to_arr, from_arr_to_data, from_arr_to_label, normalize

class Visualizer(object):


    def __init__(self, model_config, model_params, dataset_config):
        self.model_config = model_config
        self.model_params = model_params
        self.dim_data = dataset_config.input_dim
        self.dim_label = dataset_config.output_dim
        self.padding = (self.dim_data - self.dim_label) / 2
        self.normalize = dataset_config.use_preprocessing
        self.std = dataset_config.dataset_std


    def visualize(self, image_path, batch_size, best_trade_off=0.1):
        dataset, dim = self.create_data_from_image(image_path)

        compute_output = util.create_predictor(dataset, self.model_config, self.model_params, batch_size)
        predictions, labels = util.batch_predict(compute_output, dataset, self.dim_label, batch_size)
        image = self.combine_to_image(predictions, dim)

        #Need to have Mass_road structure TODO: argument
        dir = os.path.abspath(image_path + "../../../")
        #TODO: not the same extension for labels and data. In the case of MASS.
        file_ext = os.path.basename(image_path).split('.')[-1]
        label_ext = os.listdir(dir + "/labels/")[0].split('.')[-1]
        label_path = dir + "/labels/" + os.path.basename(image_path).split('.')[-2] + "." + label_ext
        label_image = Image.open(label_path, 'r')
        raw_image =  Image.open(image_path, 'r')
        hit_image = self._create_hit_image(image, raw_image, label_image , best_trade_off)
        return image, hit_image, raw_image, label_image


    def _create_hit_image(self, prediction_image, input_image, label_image, best_trade_off):
        label_image = label_image.convert('L')
        thresh = 255 * best_trade_off
        w, h = input_image.size
        w = int(w/self.dim_label)*self.dim_label
        h = int(h/self.dim_label)*self.dim_label
        p = self.padding

        input_image = input_image.crop((p, p, w-p, h-p))
        label_image = label_image.crop((p, p, w-p, h-p))

        pred = np.array(prediction_image)
        label = np.array(label_image)
        pixdata = input_image.load()
        print(pixdata)
        print("---- Creating hit/miss image")
        for y in xrange(input_image.size[1]):
            for x in xrange(input_image.size[0]):
                #print("pred",label[y][x])
                if(pred[y][x] > thresh or label[y][x] > thresh):
                    if pred[y][x] > thresh and label[y][x] >thresh:
                        pixdata[x, y] = (0, 255, 0)
                    elif label[y][x] > thresh:
                        pixdata[x, y] = (255, 0, 0)
                    else:
                         pixdata[x, y] = (0, 0, 255)

        return input_image

    def show_individual_predictions(self, dataset, predictions, std=1):
        print("Show each individual prediction")
        images = np.array(dataset[0].eval())
        for i in range(images.shape[0]):
            min_val = np.amin(images[i])
            img =from_arr_to_data((images[i]*std + min_val), 64)

            pred = predictions[i]
            clip_idx = pred < 0.3
            pred[clip_idx] = 0
            lab = from_arr_to_label(pred, 16)

            img.paste(lab, (24, 24), lab)
            img = img.resize((256, 256))
            img.show()
            user = raw_input('Proceed?')
            if user == 'no':
                break


    def combine_to_image(self, output_data, dim):
        print("Combine output to an image")

        vertical, horizontal = dim
        output = output_data.reshape(vertical, horizontal, self.dim_label, self.dim_label)
        temp = np.concatenate(output, axis=1)
        combined = np.concatenate(temp, axis=1)

        image = np.array(combined * 255, dtype=np.uint8)
        return Image.fromarray(image)


    def create_data_from_image(self ,image_path):
        print("Create data patches for model")
        dim = self.dim_label
        image = self.open_image(image_path)
        dp = 2* self.padding

        vertical = int((image.shape[0] - dp) / dim)
        horizontal = int((image.shape[1] - dp) / dim)
        number_of_patches = vertical * horizontal

        data = np.empty((number_of_patches, self.dim_data*self.dim_data*3), dtype=theano.config.floatX)
        label = np.empty((number_of_patches, dim*dim), dtype=theano.config.floatX)

        idx = 0
        for i in range(vertical):
            for j in range(horizontal):
                img_i = i * dim
                img_j = j * dim
                image_patch = from_rgb_to_arr(image[img_i: img_i + dim + dp, img_j: img_j + dim + dp])

                if self.normalize:
                    image_patch = normalize(image_patch, self.std)
                data[idx] = image_patch
                idx += 1

        aerial = AerialDataset()
        return aerial.shared_dataset([data, label], cast_to_int=True), (vertical, horizontal)


    def open_image(self, path):
        image = Image.open(path, 'r').convert('RGB')
        arr =  np.array(image)
        if image:
            del image
        return arr



