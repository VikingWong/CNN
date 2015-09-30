from PIL import Image
from augmenter.aerial import Creator

class Visualizer(object):

    def __init__(self, model):
        self.model = model
        self.creator = Creator()

    def temp_test(self):
        image = self.creator.create_image_data('./visualize/test.jpg')
        prediction = self.model.predict(image)
        print(prediction)