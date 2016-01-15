

import numpy as np
import theano
import theano.tensor as T
import timeit
from util import debug_input_data
import random
from SDG import sgd, rmsprop
import gui.server
from config import visual_params

class Evaluator(object):

    def __init__(self, model, dataset, params):
        self.data = dataset
        self.model = model
        self.params = params
        if(visual_params.gui_enabled):
            gui.server.start_new_job()

    def evaluate(self, epochs=10, verbose=False):
        L2_reg = self.params.l2_reg
        batch_size = self.params.batch_size
        momentum = self.params.momentum

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.imatrix('y')
        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)

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
        opt = rmsprop(self.model.params)
        updates = opt.updates(self.model.params, grads,learning_rate, momentum)


        train_set_x, train_set_y = self.data.set['train']
        self.train_model =  theano.function(
            [index, learning_rate],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.tester =  theano.function(
            [index, learning_rate],
            (output_layer.output, y, cost, output_layer.errors(y)),
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='ignore'
        )
        #TODO: Wierd to do it like this. A method that constructs all the functions? And integrate train in evaluate
        self._train(batch_size, epochs)



    def _train(self, batch_size, max_epochs):
        print('... training')


        n_train_batches = self._get_number_of_batches('train', batch_size)
        n_valid_batches = self._get_number_of_batches('validation', batch_size)
        n_test_batches = self._get_number_of_batches('test', batch_size)

        patience = self.params.initial_patience # look as this many examples regardless
        patience_increase = self.params.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.params.improvement_threshold # a relative improvement of this much is considered significant

        # go through this many minibatche before checking the network on the validation set
        validation_frequency = min(n_train_batches, patience / 2)
        learning_rate = self.params.initial_learning_rate/float(batch_size)
        print("Effective learning rate ", learning_rate)
        learning_adjustment = 30
        print(learning_adjustment)
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False


        while (epoch < max_epochs) and (not done_looping):
            epoch = epoch + 1
            if(epoch%learning_adjustment == 0):
                    print("Adjusting learning rate")
                    learning_rate *= 0.95
                    print("new learning rate", learning_rate)
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index


                if iter % 100 == 0:
                    print('training @ iter = ', iter)

                if epoch > 4 and (iter + 1) % (validation_frequency * 10) == 0:
                    #TODO: Make a better debugger. FIX THIS!!!
                    for test in range(1):
                        v = random.randint(0,batch_size-1)
                        output, y, cost, errs = self.tester(minibatch_index, learning_rate)
                        print(errs)
                        print(cost)
                        print(v)
                        img = self.data.set['train'][0][(minibatch_index*batch_size) + v].eval()
                        debug_input_data(img, output[v], 64, 16)
                        debug_input_data(img, y[v], 64, 16)

                #output, y, cost, errs = self.tester(minibatch_index, learning_rate)
                cost_ij = self.train_model(minibatch_index, learning_rate)

                if(np.isnan(cost_ij)):
                    print("cost IS NAN")

                if (iter + 1) % validation_frequency == 0:
                    if visual_params.gui_enabled:
                        gui.server.get_stop_status()
                    #output, y, cost, errs = self.tester(minibatch_index, learning_rate)
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
                        if visual_params.gui_enabled:
                            #TODO: Only test when validation is better, so move this out of inner scope.
                            gui.server.append_job_update(epoch, cost_ij, this_validation_loss/batch_size, test_score/batch_size)

                if patience <= iter:
                    done_looping = True
                    break
                if visual_params.gui_enabled and gui.server.stop:
                    done_looping = True

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

