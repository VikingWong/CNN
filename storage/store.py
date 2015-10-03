import pickle


class ParamStorage(object):

    def __init__(self, path='./storage/params'):
        self.path = path

    def load_params(self, path=None):
        #TODO: Check if params exists
        if not path:
            path = self.path

        f = open(path, 'rb');
        params = pickle.load(f, encoding='latin1')
        f.close()
        return params

    def store_params(self, params, path=None):
        #TODO: Check if params exist and if overwriting a file
        if not path:
            path = self.path

        f = open(path, 'wb')
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()