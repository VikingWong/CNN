import numpy as np
import sys, os
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("./"))
from interface.command import get_command
from printing import print_section, print_action

from storage import ParamStorage
from config import filename_params, dataset_params, pr_path, dataset_path
from augmenter.aerial import Creator
from data import AerialCurriculumDataset
import tools.util as util



'''
Create histograms of difference between prediction and label for dataset.
Allow finetuning of curriculum strategy.
'''
print_section('Generating plot of diff distribution between label and prediction')
#====== Arguments ===============================================
is_samples, samples = get_command('-samples', default="100")
samples = int(samples)

is_teacher_location, teacher_location = get_command('-teacher', default=filename_params.curriculum_teacher)

verify, stage = get_command('-verify', default="0")
stage = "stage" + stage

is_tradeoff, tradeoff = get_command('-tradeoff', default="0.5")
tradeoff = float(tradeoff)

#Dataset path. Config used if not supplied
is_alt_dataset, alt_dataset = get_command('-dataset')
if is_alt_dataset:
    dataset_path = alt_dataset
#==============================================================



store = ParamStorage()
teacher = store.load_params(path=teacher_location)
evaluate = util.create_simple_predictor(teacher['model'], teacher['params'])

if not verify:
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
else:
    aerial_data = AerialCurriculumDataset()
    data, labels = aerial_data.load_set(dataset_path, "train", stage=stage)

road_diff = []
non_road_diff = []
all_diff = []
pred_diff = []
nr_with_road = 0
nr_with_pred = 0

best_trade_off = tradeoff
nr_of_examples = data.shape[0]
for i in range(nr_of_examples):

    if(i%1000 == 0):
        print("{}%".format(i/float(nr_of_examples) * 100))

    data_sample = data[i]
    label_sample = labels[i]
    output = evaluate(np.array([data_sample]))
    output = util.create_threshold_image(output, best_trade_off)
    diff = np.sum(np.abs(output[0] - label_sample))/(dataset_params.output_dim*dataset_params.output_dim)

    has_road = not (np.max(label_sample) == 0)
    pred_has_road = not (np.max(output) == 0)
    if pred_has_road:
        nr_with_pred +=1
        if not has_road:
            pred_diff.append(diff)

    if has_road:
        nr_with_road +=1
        road_diff.append(diff)
    else:
        #print(diff)
        #print(np.max(output[0]))
        non_road_diff.append(diff)
    all_diff.append(diff)

road_arr = np.array(road_diff)
non_road_arr = np.array(non_road_diff)
all_arr = np.array(all_diff)
pred_arr = np.array(pred_diff)

print("Road diff mean: {}".format(np.average(road_arr)))
print("Non Road diff mean: {}".format(np.average(non_road_arr)))
print("All diff mean: {}".format(np.average(all_arr)))
print("")
print("Percentage roads: {}".format(nr_with_road/float(nr_of_examples)*100))
print("Percentage pred: {}".format(nr_with_pred/float(nr_of_examples)*100))

if not verify:
    del creator
del evaluate

font = {'size'   : 15}
matplotlib.rc('font', **font)
plt.ylabel('% of examples')
plt.xlabel('Example difficulty estimate')

#Road difficulty distribution
results, edges = np.histogram(road_arr, 50, normed=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, color='green')
plt.savefig('1.png')
plt.show()

plt.ylabel('% of examples')
plt.xlabel('Example difficulty estimate')
#Non-road difficulty distribution
results, edges = np.histogram(non_road_arr, 50, normed=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, color='red')
plt.savefig('2.png')
plt.show()

plt.ylabel('% of examples')
plt.xlabel('Example difficulty estimate')
#Combined road and non-road difficulty distribution
results, edges = np.histogram(all_arr, 50, normed=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, color='blue')
plt.savefig('3.png')
plt.show()
