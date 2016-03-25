import numpy as np
import sys, os
import theano.tensor as T
import theano
sys.path.append(os.path.abspath("./"))
from elements.custom import OutputLayer


def loss_bootstrapping(output, y, factor=1.0, size=256):
        #Customized categorical cross entropy.
        #Based on the multibox impl. More tuned to paper.
        p = output
        hard = T.gt(p, 0.5)
        #hard = p
        inter1 = ((factor * y) + ((1.0- factor) * hard)) * T.log(p)
        inter2 = ((factor * (1.0 - y)) + ((1.0- factor) * (1.0 - hard))) * T.log(1.0 - p)
        loss = (
            - T.sum( ((factor * y) + ((1.0- factor) * hard)) * T.log(p) ) -
            T.sum( ((factor * (1.0 - y)) + ((1.0- factor) * (1.0 - hard))) * T.log(1.0 - p) )
        )
        return loss/size, hard, p, loss, y, inter1, inter2

index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.imatrix('y') #label data

pred_arr = [[0.39, 0.11]]
label_arr = [[1, 0]]
s = 2
bo = loss_bootstrapping(x, y, size=s, factor=0)

#pred = np.ones((1, 256), dtype=theano.config.floatX)
#label = np.ones((1, 256), dtype=theano.config.floatX)
pred = np.array(pred_arr, dtype=theano.config.floatX)
label = np.array(label_arr, dtype=theano.config.floatX)
shared_x = theano.shared(pred, borrow=True)
shared_y = theano.shared(label, borrow=True)
casted = T.cast(shared_y, 'int32')
func = theano.function([], bo, givens={x: shared_x, y: casted})


loss, gt, output,loss2, y, i1, i2 =func()
print(np.array(output))
print(gt)
print(y)
print(loss)
print(i1, i2)
