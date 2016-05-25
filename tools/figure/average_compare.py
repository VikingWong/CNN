import sys, os

sys.path.append(os.path.abspath("./"))

import util

'''
This tool creates a two figures. The first figure plots the precision and recall data. THe other plots the MSE loss.
These plots show comparisons between different folders of results. IE, baseline and test results.
'''

sub_folder = '4'
path = '/home/olav/Documents/Results/E8'
folders = ['baseline', 'bootstrapping', 'confident bootstrapping']
pr_key_x = 'threshold'
pr_key_y = 'curve'
lc_key_x = 'epoch'
lc_key_y = 'test_loss'


print("Creating comparison figures")
all_tests = []
data = {}
nr_tests = 0
for folder in folders:
    paths = os.listdir(os.path.join(path, folder, sub_folder))
    nr_tests += len(paths)
    print("Folder", folder, "length", len(paths))
    all_tests.append(paths)
    data[folder] = []

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
    breakeven_points = util.find_breakeven(pr_avg, samples=30)
    print(folder, breakeven_points)
    series.append({"name": folder, "data": pr_avg, "breakeven": breakeven_points})
util.display_precision_recall_plot(series)

series = []
for folder in folders:
    loss_avg = util.average(data[folder], 'events', lc_key_x)
    print(folder, loss_avg[-1]['test_loss'])
    series.append({"name": folder, "data": loss_avg, "y_key": lc_key_y})
util.display_loss_curve_plot(series)