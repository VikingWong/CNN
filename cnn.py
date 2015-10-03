from evaluator import Evaluator
from model import Model
from data import MnistDataset, AerialDataset
from storage.store import ParamStorage

import pickle

#Where the magic happens
d = AerialDataset()
d.load('C:/Users/olav/Pictures/dataset2') #Input stage
m = Model([32, 128]) #Create network stage
e = Evaluator(m, d)
e.evaluate(epochs=1)

#TODO: Move to storeEngine class or similar
#Stores the model params. Model can later be restored.
p = ParamStorage(path='./storage/params')
p.store_params(m.params)

