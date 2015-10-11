from PIL import Image
import theano.tensor as T
import numpy as np
import theano

from model import Model
from storage.store import ParamStorage
from augmenter.aerial import Creator
from cnn import dataset_params


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
        image = self.combine_to_image(output, dim)
        #TODO: Create model
        #TODO: Create predict set for model
        #TODO: Run set through model
        #TODO: Recreate image from predictions.
        return None

    def build_model(self, data, number):
        x = T.matrix('x')
        shared_x = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
        self.model.build(x,number, init_params=self.params)
        return x, shared_x

    def combine_to_image(self, output_data, dim):
        output = output_data.reshape(output_data.shape[0], Visualizer.LABEL_SIZE, Visualizer.LABEL_SIZE)
        return None

    def create_data_from_image(self):
        image = self.open_image('test.jpg')
        label_size = Visualizer.LABEL_SIZE
        img_size = 256
        padding = 24
        data = []

        d = (0,0)
        for i in range(padding, image.shape[0]-padding, label_size):
            for j in range(padding, image.shape[1]-padding, label_size):

                temp = image[i- padding: i+Visualizer.IMAGE_SIZE -padding, j-padding:j+Visualizer.IMAGE_SIZE-padding]
                print('i', i , ' j', j)
                image_data = self.to_data(temp)
                data.append(image_data)

            d = (i, j)
        data = np.array(data)
        data = data.reshape(data.shape[0], Visualizer.IMAGE_SIZE * Visualizer.IMAGE_SIZE)
        return data, d

    def temp_test(self):
        x = T.matrix('x')

        #Open image to an array
        image = self.create_image_data('test.jpg')
        label_size = Visualizer.LABEL_SIZE
        imgs = []

        #Creatae subimages to predict image.
        for i in range(0, image.shape[0], label_size):
            if(i+Visualizer.IMAGE_SIZE > 256):
                break
            t = image[i: i+Visualizer.IMAGE_SIZE, 50:50+Visualizer.IMAGE_SIZE]
            print(i+Visualizer.IMAGE_SIZE)
            inp = self.to_data(t)
            imgs.append(inp)

        number = len(imgs)
        print(number)
        data = np.array(imgs)
        ddim = data.shape
        data = data.reshape(ddim[0], ddim[1]*ddim[2])
        shared_x = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
        m.build(x,number, init_params=params)
        #TODO: SO MESSY ITS NOT FUNNY
        predict = self.model.create_predict_function(x, shared_x)
        p = predict()
        output = p.reshape(number, Visualizer.LABEL_SIZE, Visualizer.LABEL_SIZE)

        fullImage = output[0]
        for i in range(1, number):
            fullImage= np.concatenate((fullImage,output[i]), axis=1)
        test = self.to_image(fullImage, number)
        ttt = Image.fromarray(test)
        ttt.show()

    def to_image(self, arr, n):
        arr = arr.reshape(Visualizer.LABEL_SIZE, Visualizer.LABEL_SIZE*n)
        arr = arr* 255
        return np.array(arr, dtype=np.uint8)

    def to_data(self, arr):
        arr = np.rollaxis(arr, 2, 0)
        arr = arr.reshape(3, arr.shape[1]* arr.shape[2])
        return arr

    def _floatX(self, d):
        #Creates a data representation suitable for GPU
        return np.asarray(d, dtype="float32")

    def open_image(self, path):
        image = Image.open(path, 'r')
        arr =  np.asarray(image, dtype="float32") / 255
        image.close()
        return arr

#TODO: Param file should also store model configuration.
store = ParamStorage()
params = store.load_params(path="../storage/params")
m = Model([64, 112])

v = Visualizer(m, params)
print("visualize")
v.visualize()