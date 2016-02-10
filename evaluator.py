

import numpy as np
import theano
import theano.tensor as T
from util import debug_input_data, show_debug_sample
from printing import print_section, print_test, print_valid
import random, sys, timeit
from sdg import Backpropagation
import gui.server
from config import visual_params
from wrapper import create_theano_func, create_profiler_func

class Evaluator(object):


    def __init__(self, model, dataset, params):
        self.data = dataset
        self.model = model
        self.params = params
        self.report = {}
        if(visual_params.gui_enabled):
            gui.server.start_new_job()


    def run(self, epochs=10, verbose=False):
        batch_size = self.params.batch_size
        self.nr_train_batches = self.data.get_total_number_of_batches(batch_size)
        self.nr_valid_batches = self._get_number_of_batches('validation', batch_size)
        self.nr_test_batches = self._get_number_of_batches('test', batch_size)
        self._build(batch_size)
        self._train(batch_size, epochs)


    def _build(self, batch_size):
        print_section('Building model')

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.imatrix('y') #label data

        #Drop switch. Only train should drop units. For testing and validation all units should be used (but output rescaled)
        drop = T.iscalar('drop')
        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)

        self.model.build(x, drop, batch_size)
        errors = self.model.get_output_layer().errors(y)

        self.test_model = create_theano_func('test', self.data, x, y, drop, [index], errors, batch_size)
        self.validate_model = create_theano_func('validation', self.data, x, y, drop, [index], errors, batch_size)

        cost = self.model.get_cost(y) + (self.params.l2_reg * self.model.getL2())
        opt = Backpropagation.create(self.model.params)
        grads = T.grad(cost, self.model.params)
        updates = opt.updates(self.model.params, grads, learning_rate, self.params.momentum)

        self.train_model = create_theano_func('train', self.data, x, y, drop, [index, learning_rate], cost, batch_size, updates=updates, dropping=True)

        self.tester = create_profiler_func(self.data, x, y, drop, [index], self.model.get_output_layer(), cost, batch_size)


    def _debug(self, batch_size, nr_batches):
        '''
        When gui has requested a debug. A random minibatch is chosen, and a number of images are displayed,
        so user can evaulate progress.
        '''
        data = []
        labels = []
        predictions = []
        number_of_tests = 6
        for test in range(number_of_tests):
            minibatch_index = random.randint(0, nr_batches-1)
            v = random.randint(0,batch_size-1)
            output, y, cost, errs = self.tester(minibatch_index)
            predictions.append(output[v])
            labels.append(y[v])
            data.append(self.data.set['train'][0][(minibatch_index*batch_size) + v].eval())

        show_debug_sample(data, labels, predictions, 64, 16, std=self.data.std)


    def _get_validation_score(self, batch_size, epoch, minibatch_index):
        validation_loss = np.mean( [self.validate_model(i) for i in range(self.nr_valid_batches)] )
        print_valid(epoch, minibatch_index + 1, self.nr_train_batches,  validation_loss)
        return validation_loss


    def _get_test_score(self, batch_size):
        test_score =  np.mean( [self.test_model(i) for i in range(self.nr_test_batches)] )
        print_test(test_score)
        return test_score


    def _get_number_of_batches(self, set_name, batch_size):
        set_x, set_y = self.data.set[set_name]
        nr_of_batches = set_x.get_value(borrow=True).shape[0]
        nr_of_batches /= batch_size
        return int(nr_of_batches)


    def _train(self, batch_size, max_epochs):
        print_section('Training model')

        patience = self.params.initial_patience # look as this many examples regardless
        patience_increase = self.params.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.params.improvement_threshold # a relative improvement of this much is considered significant

        # go through this many minibatch before checking the network on the validation set
        gui_frequency = 500
        validation_frequency = min(self.nr_train_batches, patience / 2)
        learning_rate = self.params.initial_learning_rate
        print('Effective learning rate {}'.format(learning_rate))
        learning_adjustment = self.params.epoch_learning_adjustment
        nr_learning_adjustments = 0
        learning_decrease = self.params.learning_rate_decrease

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        self.start_time = timeit.default_timer()

        nr_chunks = self.data.get_chunk_number()
        epoch = 0
        done_looping = False
        iter = 0

        #==== INITIAL PERFORMANCE ====
        validation_score = self._get_validation_score(batch_size, epoch, 0)
        test_score = self._get_test_score(batch_size)

        #==== UPDATE GUI ====
        if visual_params.gui_enabled:
                gui.server.append_job_update(epoch, None, validation_score, test_score)

        try:
            while (epoch < max_epochs) and (not done_looping):
                epoch = epoch + 1
                if(epoch % learning_adjustment == 0):
                        learning_rate *= learning_decrease
                        nr_learning_adjustments += 1
                        print('---- New learning rate {}'.format(learning_rate))

                #For current examples chunk in GPU memory
                for chunk_index in range(nr_chunks):
                    self.data.switch_active_training_set( chunk_index )
                    nr_elements = self.data.get_elements( chunk_index )
                    chunk_batches = nr_elements / batch_size

                    #Each chunk contains a certain number of batches.
                    for minibatch_index in range(chunk_batches):
                        cost_ij = self.train_model(minibatch_index, learning_rate)

                        if iter % 1000 == 0:
                            print('---- Training @ iter = {}. Patience = {}'.format(iter, patience))

                        if visual_params.gui_enabled and iter % gui_frequency == 0:
                            gui.server.get_command_status()

                        if visual_params.gui_enabled and (iter - 100) % gui_frequency ==0 and gui.server.is_testing():
                            self._debug(batch_size, chunk_batches)

                        if(np.isnan(cost_ij)):
                            print('cost IS NAN')

                        #==== EVAULATE ====
                        if (iter + 1) % validation_frequency == 0:

                            #==== CURRENT PERFORMANCE ====
                            validation_score = self._get_validation_score(batch_size, epoch, minibatch_index)
                            test_score = self._get_test_score(batch_size)

                            #==== UPDATE GUI ====
                            if visual_params.gui_enabled:
                                    gui.server.append_job_update(epoch, cost_ij, validation_score, test_score)

                            #==== EARLY STOPPING ====
                            if validation_score < best_validation_loss:

                                #improve patience if loss improvement is good enough
                                if validation_score < best_validation_loss * improvement_threshold:
                                    patience = max(patience, iter * patience_increase)
                                    print("---- New best validation loss. Patience increased to {}".format(patience))

                                # save best validation score and iteration number
                                best_validation_loss = validation_score
                                best_iter = iter

                        if patience <= iter:
                            done_looping = True
                            break
                        if visual_params.gui_enabled and gui.server.stop:
                            done_looping = True

                        iter += 1 #Increment interation after each batch has been processed.

        except KeyboardInterrupt:
            self.set_result(best_iter, iter, best_validation_loss, test_score, nr_learning_adjustments, epoch)
            print("Inpterupted by user. Current model params will be saved now.")
        except Exception as e:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        self.set_result(best_iter, iter, best_validation_loss, test_score, nr_learning_adjustments, epoch)


    def set_result(self, best_iter, iter, valid, test, nr_learning_adjustments, epoch):
        end_time = timeit.default_timer()
        duration = (end_time - self.start_time) / 60.
        valid_end_score = valid
        test_end_score = test
        print('Optimization complete.')
        print('Best validation score of %f obtained at iteration %i, '
              'with test performance %f' %
              (valid_end_score, best_iter + 1, test_end_score))
        print('The code ran for %.2fm' % (duration))

        self.report['evaluation'] = {
            'best_iteration': best_iter+1, 'iteration': iter, 'test_score': test_end_score, 'valid_score': valid_end_score,
            'learning_adjustments': nr_learning_adjustments, 'epoch': epoch, 'duration': duration
        }
        self.report['dataset'] = self.data.get_report()


    def get_result(self):
        return self.report