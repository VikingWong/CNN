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

        dim = (self.dataset_config.input_dim, self.dataset_config.output_dim)
        preprocessing = self.dataset_config.use_preprocessing
        print("---- Using preprossing: {}".format(preprocessing))
        std = self.dataset_config.dataset_std
        self.creator = Creator(self.dataset_path, dim=dim, preproccessing=preprocessing, std=std)
        self.creator.load_dataset()


    def create_dataset(self, is_baseline):
        print("---- Starting sampling. WARNING: this might take a while.")
        base_sampling = self.dataset_config.samples_per_image
        self._generate_stage("main", 0.1, base_sampling)

    def _generate_stage(self, name, threshold, samples):
        #TODO: generate folder, create dataset, then store samples in directory of NAME.
        self.creator.sample_data(self.creator.train, samples, curriculum=self.evaluate, curriculum_threshold=threshold)


    #a = np.array([0,1,2,3])
    #a.dump("./tools/curriculum/file.npy")

    #b = np.load("./tools/curriculum/file.npy")

    #print(b)