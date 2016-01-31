import pickle
from util import Params

class ParamStorage(object):

    def __init__(self, path='./storage/params.pkl'):
        self.path = path

    def load_params(self, path=None):
        #TODO: Check if params exists
        if not path:
            path = self.path

        f = open(path, 'rb');
        params = pickle.load(f)
        f.close()
        return params

    def store_params(self, params, model, dataset, optimization, epochs, path=None):
        #TODO: Check if params exist and if overwriting a file
        if not path:
            path = self.path

        data = {'params': params, 'model': model, 'dataset': dataset, 'optimization': optimization, 'epochs': epochs}
        print(path)
        f = open(path, 'wb')
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()