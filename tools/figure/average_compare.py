import sys, os

sys.path.append(os.path.abspath("./"))

import util

sub_folder = ''
path = '/home/olav/Documents/Results/anticurr100'
folders = ['baseline','curriculum','anti-curriculum' ]
pr_key_x = 'threshold'
pr_key_y = 'valid_curve'
lc_key_x = 'epoch'
lc_key_y = 'validation_loss'


print("Creating comparison figures")
baseline_paths = os.listdir(os.path.join(path, folders[0], sub_folder))
test_paths = os.listdir(os.path.join(path, folders[1], sub_folder))
curriculum_paths = os.listdir(os.path.join(path, folders[2], sub_folder))
all_tests = [baseline_paths, test_paths, curriculum_paths]

data = {folders[0]: [], folders[1]: [], folders[2]: []}
print("length", len(all_tests[0]), len(all_tests[1]), len(all_tests[2]) )
for t in range(len(all_tests)):
    for data_path in all_tests[t]:
        json_data = util.open_json_result(os.path.join(path, folders[t], sub_folder, data_path))

        if type(json_data) is list:
            d = json_data[0]
        else:
            d = json_data
        data[folders[t]].append(d)

series = []
for folder in folders:
    pr_avg = util.average(data[folder], pr_key_y, pr_key_x)
    series.append({"name": folder, "data": pr_avg})
util.display_precision_recall_plot(series)

series = []
for folder in folders:
    loss_avg = util.average(data[folder], 'events', lc_key_x)
    series.append({"name": folder, "data": loss_avg, "y_key": lc_key_y})
util.display_loss_curve_plot(series)