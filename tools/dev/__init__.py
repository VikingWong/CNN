import numpy as np
import os, sys, random

#If the stage switching replace training set, by picking index with replacement. The pool of indices does not diminish,
#the simple dataset will then be replaced by the exponential of iterations of non-replaced examples
iterations = 20
data = np.ones(80000)
prev_data = np.zeros(400000)
elements = prev_data.shape[0]
for j in range(iterations):
    for i in range(data.shape[0]):
        nx = random.randint(0, elements-1)
        prev_data[nx] = data[i]

per = np.sum(prev_data)/(elements)*100
print("Replaced percent of examples after " + str(iterations) + ": " + str(per))
print(100-per)