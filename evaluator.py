__author__ = 'olav'

from model import Model
import numpy as np
import theano
import theano.tensor as T
import timeit

#TODO: Generalized evaluator. Contains basic SGD, and utilize a model object where
#TODO: model specific things recide.
class Evaluator(object):
    PATIENCE = 10000        # look as this many examples regardless
    PATIENCE_INCREASE = 2   # wait this much longer when a new best is found
    IMPROVEMENT_THRESHOLD = 0.995 # a relative improvement of this much is considered significant
    NUMBER_OF_KERNELS = [64, 256]

    def __init__(self, model, dataset):
        self.data = dataset
        self.model = model

    def evaluate(self, learning_rate=0.1, epochs=200, batch_size=64):

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.imatrix('y')

        self.model.build(x, batch_size)
        output_layer = self.model.get_output_layer()
        cost = self.model.get_cost(y)
        errors = self.model.get_errors(y)
        #Maybe model methods. x and y kept in model?
        self.test_model = self._create_test_function(x,y,batch_size, index,cost, output_layer.errors(y))

        self.validate_model = self._create_validation_function(x,y,batch_size, index,cost,output_layer.errors(y))
        grads = T.grad(cost, self.model.params)

        updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(self.model.params, grads)
        ]

        self.train_model = self._create_training_function(x, y, batch_size, index, cost, updates)

        self._train(batch_size, epochs)



    def _train(self, batch_size, max_epochs):
        print('... training')

        n_train_batches = self._get_number_of_batches('train', batch_size)
        n_valid_batches = self._get_number_of_batches('validation', batch_size)
        n_test_batches = self._get_number_of_batches('test', batch_size)

        patience = Evaluator.PATIENCE
        patience_increase = Evaluator.PATIENCE_INCREASE
        improvement_threshold = Evaluator.IMPROVEMENT_THRESHOLD
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
                print(minibatch_index)
                print(self.data.set['train'][0].shape.eval())
                print(self.data.set['train'][1].shape.eval())
                cost_ij = self.train_model(minibatch_index)
                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

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
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    def _create_training_function(self, x,y, batch_size, index, cost, updates):
        train_set_x, train_set_y = self.data.set['train']
        return theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def _create_test_function(self, x,y, batch_size, index, cost, errors):
        test_set_x, test_set_y = self.data.set['test']
        return theano.function(
            [index],
            errors,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def _create_validation_function(self, x,y, batch_size, index, cost, errors):
        valid_set_x, valid_set_y = self.data.set['validation']
        return theano.function(
            [index],
            errors,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def _get_number_of_batches(self, set_name, batch_size):
        set_x, set_y = self.data.set[set_name]
        nr_of_batches = set_x.get_value(borrow=True).shape[0]
        nr_of_batches /= batch_size
        return int(nr_of_batches)

