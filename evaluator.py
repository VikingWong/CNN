

from model import Model
import numpy as np
import theano
import theano.tensor as T
import timeit
from util import debug_input_data
import random
#TODO: Generalized evaluator. Contains basic SGD, and utilize a model object where
#TODO: model specific things recide.
class Evaluator(object):

    def __init__(self, model, dataset):
        self.data = dataset
        self.model = model

    def evaluate(self, params, epochs=10, verbose=False):
        L2_reg = params.l2_reg
        learning_rate = params.initial_learning_rate
        batch_size = params.batch_size

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.imatrix('y')


        self.model.build(x, batch_size)
        output_layer = self.model.get_output_layer()
        cost = self.model.get_cost(y) + L2_reg * self.model.getL2()

        #errors = self.model.get_errors(y)
        test_set_x, test_set_y = self.data.set['test']
        self.test_model = theano.function(
            [index],
            output_layer.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        valid_set_x, valid_set_y = self.data.set['validation']
        self.validate_model =  theano.function(
            [index],
            output_layer.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        grads = T.grad(cost, self.model.params)

        updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(self.model.params, grads)
        ]


        train_set_x, train_set_y = self.data.set['train']
        self.train_model =  theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.tester =  theano.function(
            [index],
            (output_layer.output, y, cost, output_layer.errors(y)),
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='ignore'
        )

        self._train(batch_size, epochs, params)



    def _train(self, batch_size, max_epochs, params):
        print('... training')


        n_train_batches = self._get_number_of_batches('train', batch_size)
        n_valid_batches = self._get_number_of_batches('validation', batch_size)
        n_test_batches = self._get_number_of_batches('test', batch_size)

        patience = params.initial_patience # look as this many examples regardless
        patience_increase = params.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = params.improvement_threshold # a relative improvement of this much is considered significant

        # go through this many minibatche before checking the network on the validation set
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False


        while (epoch < max_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                print("IT")
                #output, y, cost, errs = self.tester(minibatch_index)
                #print("errors: ", errs)
                #print("TEMP____________")
                #print(cost)
                #print(errs)
                #print(output.shape)
                #print(y.shape)
                #print(output[0, 0: 16])
                #print(y[0, 0: 16])
                #print(T.sum(T.nnet.binary_crossentropy(output[0, 0: 256], y[0, 0: 256])).eval())
                #print(T.sum(T.nnet.binary_crossentropy(output, y)).eval())
                #print("TEMP____________")
                #raise Exception("NO MORE")
                if epoch > 150   and (iter + 1) % validation_frequency == 0:
                    #TODO: Make a better debugger.
                    for test in range(1):
                        v = random.randint(0,n_train_batches)
                        output, y, cost, errs = self.tester(v)
                        print(errs)
                        print(cost)
                        debug_input_data(self.data.set['train'][0][v].eval(), output, 64, 16)
                        debug_input_data(self.data.set['train'][0][v].eval(), y, 64, 16)

                cost_ij = self.train_model(minibatch_index)
                if (iter + 1) % validation_frequency == 0:
                    #output, y, cost, errs = self.tester(minibatch_index)
                    #print(cost)
                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f MSE' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss/batch_size))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in range(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f MSE') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score/batch_size))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print('The code ran for %.2fm' % ((end_time - start_time) / 60.))



    def _get_number_of_batches(self, set_name, batch_size):
        set_x, set_y = self.data.set[set_name]
        nr_of_batches = set_x.get_value(borrow=True).shape[0]
        nr_of_batches /= batch_size
        return int(nr_of_batches)

