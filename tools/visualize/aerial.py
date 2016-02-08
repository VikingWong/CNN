from PIL import Image
import theano.tensor as T
import numpy as np
import theano
import math, sys, os

sys.path.append(os.path.abspath("./"))

from tools.util import create_predictor


from augmenter import from_rgb_to_arr, from_arr_to_data, from_arr_to_label, normalize

class Visualizer(object):
    LABEL_SIZE = 16
    IMAGE_SIZE = 64

    def __init__(self, model, params, std= 1):
        self.model = model
        self.params = params
        self.std = std
        print("STD:", self.std)



    def visualize(self):
        data, dim = self.create_data_from_image()
        print(data.nbytes / 1000000, "mb")
        x, shared_x = self.build_model(data, data.shape[0])
        predict = self.model.create_predict_function(x, shared_x)
        output = predict()
        image = self.combine_to_image(output, dim)
        self.show_individual_predictions(data, output)
        return image


    def build_model(self, data, number):
        print("Build model and predict function")
        x = T.matrix('x')
        shared_x = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)

        self.model.build(x, number, init_params=self.params)
        return x, shared_x


    def show_individual_predictions(self, images, predictions):
        print("Show each individual prediction")
        for i in range(124, images.shape[0]):

            img =from_arr_to_data(images[i], 64)
            pred = predictions[i]
            print(i)


            clip_idx = pred < 0.3
            pred[clip_idx] = 0
            lab = from_arr_to_label(pred, 16)

            #img.paste(lab, (24, 24), lab)
            lab.show()
            img.show()
            del img
            del lab
            user = raw_input('Proceed?')
            if user == 'no':
                break


    def combine_to_image(self, output_data, dim):
        print("Combine output to an image")
        #Assume square tiles so sqrt will get dimensions.
        label_image_dim = (int)(math.sqrt(output_data.shape[0]))
        label_size = Visualizer.LABEL_SIZE
        output = output_data.reshape(label_image_dim, label_image_dim, label_size, label_size)

        #Output is a matrix of 16x16 patches. Is combined to one image below.
        l = []
        for i in range(0, label_image_dim):
            arr = output[i][0]
            for j in range(1, label_image_dim):
                arr = np.hstack((arr, output[i][j]))
            l.append(arr)

        output = np.vstack(l)

        output = output * 255
        image = np.array(output, dtype=np.uint8)
        return Image.fromarray(image)


    def create_data_from_image(self):
        print("Create data patches for model")
        image = self.open_image('/home/olav/Pictures/Mass_roads/test/data/24628885_15.tiff')
        image = image[0:1024, 0: 1024, :]
        #Need to be a multiply of 2 for now.
        label_size = Visualizer.LABEL_SIZE
        padding = 24
        data = []

        d = (image.shape[0]- (2*padding), image.shape[1] - (2 * padding))
        for i in range(padding, image.shape[0]-padding, label_size):
            for j in range(padding, image.shape[1]-padding, label_size):
                temp = image[i- padding: i+Visualizer.IMAGE_SIZE -padding, j-padding:j+Visualizer.IMAGE_SIZE-padding]
                image_data = from_rgb_to_arr(temp)

                #TODO: Store preprocessing in params file as well. Will have concequences if config is out of sync with stored values.
                if True:
                    image_data = normalize(image_data, self.std)
                data.append(image_data)

        data = np.array(data)

        return data, d


    def open_image(self, path):
        image = Image.open(path, 'r')
        arr =  np.array(image)
        if image:
            del image
        return arr



