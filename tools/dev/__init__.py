import numpy as np
import os, sys, random

data = np.ones(110800)
prev_data = np.zeros(110800)
elements = 110800
for i in range(elements):
    nx = random.randint(0, elements-1)
    prev_data[nx] = data[i]


print(np.sum(prev_data)/(elements))