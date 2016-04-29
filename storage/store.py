import pickle
from util import Params
import json, os
from config import model_params, optimization_params, dataset_params, number_of_epochs, filename_params

class ParamStorage(object):

    def __init__(self, path=None):
        if not path:
            self.path = filename_params.network_save_name
        else:
            self.path = path

    def load_params(self, path=None):
        #TODO: Check if params exists
        if not path:
            path = self.path

        f = open(path, 'rb')
        params = pickle.load(f)
        f.close()
        return params

    def store_params(self, params, path=None, id=''):
        #TODO: Check if params exist and if overwriting a file
        if not path:
            path = self.path

        data = {
            'params': params,
            'model': model_params,
            'dataset': dataset_params,
            'optimization': optimization_params,
            'epochs': number_of_epochs
        }
        print(path)
        #ID, on wrong place
        file_path = os.path.splitext(path)[0]
        extension = os.path.splitext(path)[1]
        f = open(file_path + id + extension, 'wb')
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def store_result(self, path, loss, test_pr, valid_pr):

        data = [{
            'events': loss,
            'curve': test_pr,
            'valid_curve': valid_pr,
        }]
        with open(path, 'w') as fp:
            json.dump(data, fp)