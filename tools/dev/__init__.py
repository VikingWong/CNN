import numpy as np
import os, sys, random

data = np.ones(110800)
prev_data = np.zeros(443201)
elements = prev_data.shape[0]
for i in range(data.shape[0]):
    nx = random.randint(0, elements-1)
    prev_data[nx] = data[i]


print(np.sum(prev_data)/(elements))