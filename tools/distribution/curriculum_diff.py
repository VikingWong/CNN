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
Create histograms of difference between prediction and label for dataset.
Allow finetuning of curriculum strategy.
'''
print_section('Generating plot of diff distribution between label and prediction')
threshold = 1.0

if '-t' in sys.argv:
    idx = sys.argv.index('-t')
    threshold = float(sys.argv[idx+1])
    print_action("threshold set to {}".format(threshold))

samples = 100
if '-s' in sys.argv:
    idx = sys.argv.index('-s')
    samples = int(sys.argv[idx+1])
    print_action("samples set to {}".format(samples))

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
    samples,
    rotation=dataset_params.use_rotation
)

road_diff = []
non_road_diff = []
all_diff = []
pred_diff = []

nr_of_examples = data.shape[0]
for i in range(nr_of_examples):

    if(i%1000 == 0):
        print("{}%".format(i/float(nr_of_examples) * 100))

    data_sample = data[i]
    label_sample = labels[i]
    output = evaluate(np.array([data_sample]))
    diff = np.sum(np.abs(output[0] - label_sample))/(dataset_params.output_dim*dataset_params.output_dim)

    has_road = not (np.max(label_sample) == 0)
    pred_has_road = not (np.max(output) == 0)
    if pred_has_road:
        pred_diff.append(diff)
    if(has_road):
        road_diff.append(diff)
    else:
        non_road_diff.append(diff)
    all_diff.append(diff)

road_arr = np.array(road_diff)
non_road_arr = np.array(non_road_diff)
all_arr = np.array(all_diff)
pred_arr = np.array(pred_diff)

print("Road diff mean: {}".format(np.average(road_arr)))
print("Non Road diff mean: {}".format(np.average(non_road_arr)))
print("All diff mean: {}".format(np.average(all_arr)))
print("Percentage roads: {}".format(len(road_diff)/float(len(all_diff))*100))

plt.figure(1)
plt.subplot(411)
n, bins, patches = plt.hist(road_arr, 120, normed=0, color='green')
plt.subplot(412)
n, bins, patches = plt.hist(non_road_arr, 120, normed=0, color='red')
plt.subplot(413)
n, bins, patches = plt.hist(all_arr, 120, normed=0, color='blue')
plt.subplot(414)
n, bins, patches = plt.hist(pred_arr, 120, normed=0, color='grey')
plt.show()
