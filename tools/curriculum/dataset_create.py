import numpy as np
import sys, os

sys.path.append(os.path.abspath("./"))

from augmenter import Creator
from data import AerialDataset
import tools.util as util

#TODO: Store in smaller files (Maybe)
#TODO: Calculate distribution of diffs maybe? Plot it to get at better understanding how to make stages.

class CurriculumDataset(object):

    def __init__(self, teacher, dataset_path, store_path, dataset_config):
        self.dataset_path = dataset_path
        self.store_path = store_path
        self.teacher = teacher
        self.dataset_config = dataset_config
        self.rotate = dataset_config.use_rotation

        if os.path.exists(self.store_path):
            raise Exception("Store path already exists")
        else:
            os.makedirs(self.store_path)
            os.makedirs(os.path.join(self.store_path, "train"))
            os.makedirs(os.path.join(self.store_path, "valid"))
            os.makedirs(os.path.join(self.store_path, "test"))

        self.evaluate = util.create_simple_predictor(teacher['model'], teacher['params'])
        self.creator = Creator(
            self.dataset_path,
            dim=(self.dataset_config.input_dim, self.dataset_config.output_dim),
            preproccessing=self.dataset_config.use_preprocessing,
            std=self.dataset_config.dataset_std,
            reduce_training=self.dataset_config.reduce_training,
            reduce_testing=self.dataset_config.reduce_testing,
            reduce_validation=self.dataset_config.reduce_validation
        )
        self.creator.load_dataset()


    def create_dataset(self, is_baseline):
        print("---- Starting sampling. WARNING: this might take a while.")
        base_sampling = self.dataset_config.samples_per_image
        curriculum_sampling = np.ceil(base_sampling/10)

        #Sampling at different thresholds.
        thresholds = np.arange(0.01 , 0.8, 0.05)
        if is_baseline:
            thresholds = np.ones(thresholds.shape)

        print("---- Main dataset")
        self._generate_stage("stage0", thresholds[0], base_sampling)
        for i in range(1, thresholds.shape[0]):
            print("---- Stage{} dataset".format(i))
            self._generate_stage("stage{}".format(i), thresholds[i], curriculum_sampling)

        self._generate_set("test", self.creator.test, base_sampling)
        self._generate_set("valid", self.creator.valid, base_sampling)


    def _generate_set(self, set_name, dataset, samples):
        '''
        Validation and test data is also pre-generated. This means the result is self contained.
        '''
        data, labels = self.creator.sample_data(dataset, samples)
        stage_path = os.path.join(self.store_path, set_name)
        os.makedirs(os.path.join(stage_path, "labels"))
        os.makedirs(os.path.join(stage_path, "data"))
        np.save(os.path.join(stage_path, "labels", "examples"), labels)
        np.save(os.path.join(stage_path, "data", "examples"), data)


    def _generate_stage(self, name, threshold, samples):
        '''
        Training set is a special case, which involve training folder with several stages. These
         stages can be introduced in the active training data over time. Slowly transforming the simple distribution
         to the real dataset distribution of data.
        :return:
        '''
        stage_path = os.path.join(self.store_path, "train", name)
        os.makedirs(stage_path)
        data, labels = self.creator.sample_data(
            self.creator.train,
            samples,
            curriculum=self.evaluate,
            curriculum_threshold=threshold,
            rotation=self.rotate
        )
        os.makedirs(os.path.join(stage_path, "labels"))
        os.makedirs(os.path.join(stage_path, "data"))
        np.save(os.path.join(stage_path, "labels", "examples"), labels)
        np.save(os.path.join(stage_path, "data", "examples"), data)
