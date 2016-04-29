import sys, os

sys.path.append(os.path.abspath("./"))

import util

'''
This tool creates a breakeven/loss over epoch plot, showing the relationship between relaxed precision and recall and
MSE test loss.
'''

sub_folder = ''
path = '/home/olav/Documents/Results/E7_inexperienced_teacher/pr-per-epoch'
folders = [ 'test loss', 'relaxed precision and recall', 'precise precision and recall']
pr_key_x = 'threshold'
pr_key_y = 'curve'
lc_key_x = 'epoch'
lc_key_y = 'test_loss'
pr_epoch = 5

def sorter(x):
    number = x[2:].split('.')[0]
    return int(number)

print("Creating comparison figures")
all_tests = []
data = {}
nr_tests = 0
for folder in folders:
    paths = os.listdir(os.path.join(path, folder, sub_folder))
    nr_tests += len(paths)
    print("Folder", folder, "length", len(paths))
    if folder != folders[0]:
       paths.sort(key=sorter )
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

manual_breakeven = [[0.513, 0.595, 0.687, 0.712], [0.337, 0.411, 0.462, 0.524]] #Very uneven curves is hard to approximate by polyfit. (finds breakeven automatically)
compare_series = []
for j, pr in enumerate(folders[1:]):
    pr_per_epoch = []
    for i, curve in enumerate(data[pr]):
        samples = 10
        if  i<len(manual_breakeven[j]):
            breakeven_points = [0, manual_breakeven[j][i]]
        else:
            breakeven_points = util.find_breakeven(curve[pr_key_y], samples=samples)
        print(folder, breakeven_points)
        name = (pr_epoch * i) +5
        series.append({"name": "Epoch " + str(name), "data": curve[pr_key_y], "breakeven": breakeven_points})
        pr_per_epoch.append({'epoch': name, 'breakeven': breakeven_points[-1]})
    compare_series.append({'name': pr, "data": pr_per_epoch, 'y_key': 'breakeven'})
util.display_precision_recall_plot(series)

series = []
loss_avg = util.average(data[folders[0]], 'events', lc_key_x)
series.append({"name": folders[0], "data": loss_avg, "y_key": lc_key_y})
util.display_two_axis_plot(series, compare_series)