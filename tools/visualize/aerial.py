from PIL import Image
import theano.tensor as T
import numpy as np
import theano
import math

from model import Model
from storage.store import ParamStorage
from augmenter.aerial import Creator
from config import model_params
from util import from_rgb_to_arr, from_arr_to_data, from_arr_to_label

class Visualizer(object):
    LABEL_SIZE = 16
    IMAGE_SIZE = 64

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.creator = Creator()


    def visualize(self):
        data, dim = self.create_data_from_image()
        x, shared_x = self.build_model(data, data.shape[0])
        predict = self.model.create_predict_function(x, shared_x)
        output = predict()
        print(output.shape)
        image = self.combine_to_image(output)
        #self.show_individual_predictions(data, output)
        return image


    def build_model(self, data, number):
        x = T.matrix('x')
        shared_x = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
        self.model.build(x,number, init_params=self.params)
        return x, shared_x


    def show_individual_predictions(self, images, predictions):
        print(images.shape)
        for i in range(images.shape[0]):

            img =from_arr_to_data(images[i], 64)
            print(img)
            pred = predictions[i]
            clip_idx = pred < 0.3
            pred[clip_idx] = 0
            lab = from_arr_to_label(pred, 16)

            img.paste(lab, (24, 24), lab)
            img.show()
            img.close()
            lab.close()
            user = input('Proceed?')
            if user == 'no':
                break




    def combine_to_image(self, output_data):
        print(output_data.shape)
        label_size = Visualizer.LABEL_SIZE
        output = output_data.reshape(math.sqrt(output_data.shape[0])*label_size,math.sqrt(output_data.shape[0])*label_size)
        output = output * 255
        image = np.array(output, dtype=np.uint8)
        return Image.fromarray(image)


    def create_data_from_image(self):
        image = self.open_image('test2.tiff')
        image = image[0:1472, 0: 1472, :]
        #Need to be a multiply of 2 for now.
        label_size = Visualizer.LABEL_SIZE
        print(image.shape)
        img_size = image.shape[0]
        print(img_size)
        padding = 24
        data = []

        d = (image.shape[0]- (2*padding), image.shape[1] - (2 * padding))
        for i in range(padding, image.shape[0]-padding, label_size):
            for j in range(padding, image.shape[1]-padding, label_size):
                temp = image[i- padding: i+Visualizer.IMAGE_SIZE -padding, j-padding:j+Visualizer.IMAGE_SIZE-padding]
                image_data = from_rgb_to_arr(temp)
                data.append(image_data)

        data = np.array(data)

        return data, d


    def open_image(self, path):
        image = Image.open(path, 'r')
        arr =  np.array(image)
        image.close()
        return arr

#TODO: Param file should also store model configuration.
print("======================================")
store = ParamStorage()
params = store.load_params(path="../../results/params.pkl")
m = Model(model_params)
print(params)
v = Visualizer(m, params)
print("visualize")
img = v.visualize()
img.show()
img.save('./tester.jpg')