
import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("./"))
from printing import print_section, print_action
from storage import ParamStorage
from config import filename_params, dataset_params, pr_path
from augmenter import Creator
import tools.util as util

'''
Use sampler and curriculum to create distribution of diffs
    -For all
    -For road diffs
    -For non road diffs

Use this to determine how to create a curriculum learning scheme.
'''
print_section('Generating plot of diff distribution between label and prediction')
threshold = 1.0

if '-t' in sys.argv:
    idx = sys.argv.index('-t')
    threshold = float(sys.argv[idx+1])
    print_action("threshold set to {}".format(threshold))
    
store = ParamStorage()
teacher = store.load_params(path=filename_params.curriculum_teacher)
evaluate = util.create_simple_predictor(teacher['model'], teacher['params'])

creator = Creator(
    pr_path,
    dim=(dataset_params.input_dim, dataset_params.output_dim),
    preproccessing=dataset_params.use_preprocessing,
    std=dataset_params.dataset_std,
    reduce_training=dataset_params.reduce_training,
    reduce_testing=dataset_params.reduce_testing,
    reduce_validation=dataset_params.reduce_validation
)
creator.load_dataset()

data, labels = creator.sample_data(
    creator.train,
    threshold,
    rotation=dataset_params.use_rotation
)

road_diff = []
non_road_diff = []
nr_of_examples = data.shape[0]
for i in range(nr_of_examples):

    if(i%1000 == 0):
        print("{}%".format(i/float(nr_of_examples) * 100))

    data_sample = data[i]
    label_sample = labels[i]
    output = evaluate(np.array([data_sample]))
    diff = np.sum(np.abs(output[0] - label_sample))/(dataset_params.output_dim*dataset_params.output_dim)

    has_road = not (np.max(label_sample) == 0)
    if(has_road):
        road_diff.append(diff)
    else:
        non_road_diff.append(diff)

road_arr = np.array(road_diff)
non_road_arr = np.array(non_road_diff)
print("Road diff mean: {}".format(np.average(road_arr)))
print("Non Road diff mean: {}".format(np.average(non_road_arr)))
plt.figure(1)
plt.subplot(121)
n, bins, patches = plt.hist(road_arr, 100, normed=1, color='green')
plt.subplot(122)
n, bins, patches = plt.hist(non_road_arr, 100, normed=1, color='red')
plt.show()
