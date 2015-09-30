__author__ = 'olav'
from evaluator import Evaluator
from model import Model
from data import MnistDataset, AerialDataset
from visualize.aerial import Visualizer

#Where the magic happens
d = AerialDataset()
d.load('C:/Users/olav/Pictures/dataset2') #Input stage
m = Model([32, 128]) #Create network stage
e = Evaluator(m, d)
e.evaluate(epochs=1)

#Test on image
v = Visualizer(m)
v.temp_test()