__author__ = 'olav'
from evaluator import Evaluator
from model import Model
from data import Dataset

#Where the magic happens
d = Dataset()
d.load('../theano-learning/deeplearning/mnist.pkl.gz')
m = Model([20, 50])
e = Evaluator(m, d)
e.evaluate()