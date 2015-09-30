__author__ = 'olav'
from evaluator import Evaluator
from model import Model
from data import MnistDataset, AerialDataset

#Where the magic happens
d = AerialDataset()
d.load('C:/Users/Olav/Pictures/dataset2') #Input stage
m = Model([64, 256]) #Create network stage
e = Evaluator(m, d)
e.evaluate()