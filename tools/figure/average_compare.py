import sys, os

sys.path.append(os.path.abspath("./"))

import util

path = '/home/olav/Documents/Results/curr100/'
folders = ['baseline', 'curriculum']
pr_key_x = 'threshold'
lc_key_x = 'epoch'
lc_key_y = 'training_loss'

print("Creating comparison figures")
baseline_paths = os.listdir(os.path.join(path, folders[0]))
test_paths = os.listdir(os.path.join(path, folders[1]))
all_tests = [baseline_paths, test_paths]

data = {folders[0]: [], folders[1]: []}

for t in range(len(all_tests)):
    for data_path in all_tests[t]:
        json_data = util.open_json_result(os.path.join(path, folders[t],data_path))
        data[folders[t]].append(json_data[0])


pr_avg_baseline = util.average(data[folders[0]], 'curve', pr_key_x)
pr_avg_test = util.average(data[folders[1]], 'curve', pr_key_x)
series = [{"name": folders[0], "data": pr_avg_baseline}, {"name": folders[1], "data": pr_avg_test}]
util.display_precision_recall_plot(series)

loss_avg_baseline = util.average(data[folders[0]], 'events', lc_key_x)
loss_avg_test = util.average(data[folders[1]], 'events', lc_key_x)
series = [
    {"name": folders[0], "data": loss_avg_baseline, "y_key": lc_key_y},
    {"name": folders[1], "data": loss_avg_test, "y_key": lc_key_y}
]
util.display_loss_curve_plot(series)