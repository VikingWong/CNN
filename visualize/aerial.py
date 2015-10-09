from PIL import Image
from model import Model
from storage.store import ParamStorage
import theano.tensor as T
import numpy as np
from augmenter.aerial import Creator
import theano

class Visualizer(object):
    LABEL_SIZE = 16
    IMAGE_SIZE = 64
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.creator = Creator()

    def temp_test(self):
        x = T.matrix('x')
        x.tag.test_value = np.random.rand(5, 10)

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
        output = p.reshape(number, 16, 16)

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

    def create_image_data(self, path):
        image = Image.open(path, 'r')
        arr =  np.asarray(image, dtype="float32") / 255
        image.close()
        return arr

store = ParamStorage()
params = store.load_params(path="../storage/params")
m = Model([64, 112])
v = Visualizer(m, params)
v.temp_test()