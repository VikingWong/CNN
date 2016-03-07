import numpy as np
import sys, os

sys.path.append(os.path.abspath("./"))

from augmenter import Creator
from data import AerialDataset
import tools.util as util


class CurriculumDataset(object):

    def __init__(self, teacher, dataset_path, store_path, dataset_config):
        self.dataset_path = dataset_path
        self.store_path = store_path
        self.teacher = teacher
        self.dataset_config = dataset_config
        self.evaluate = util.create_simple_predictor(teacher['model'], teacher['params'])


    def create_dataset(self):
        dim = (self.dataset_config.input_dim, self.dataset_config.output_dim)
        path = self.dataset_path
        preprocessing = self.dataset_config.use_preprocessing
        print("---- Using preprossing: {}".format(preprocessing))
        std = self.dataset_config.dataset_std
        samples_per_image = 400 #TODO: USE CONFIG
        creator = Creator(path, dim=dim, preproccessing=preprocessing, std=std)
        creator.load_dataset()
        creator.sample_data(creator.test, samples_per_image, curriculum=self.evaluate)



    #a = np.array([0,1,2,3])
    #a.dump("./tools/curriculum/file.npy")

    #b = np.load("./tools/curriculum/file.npy")

    #print(b)