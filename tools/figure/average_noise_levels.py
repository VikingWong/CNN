import sys, os

sys.path.append(os.path.abspath("./"))

import util

'''
This tool creates a figure showing, breakeven and final loss over several levels of noise.
Compact representation of experiments conducted for several levels of noise.
Assume folders, contains sub-folders, with experimental results. Enter mapping between subfolder and noise percentage
in sub_folders variable.
'''

path = '/home/olav/Documents/Results/E1-mass-boot-100'
folders = [ 'baseline', 'bootstrapping']
sub_folders = [{'name': '0', 'value': 0.0}, {'name': '1', 'value': 0.1},{'name': '2', 'value': 0.2},
               {'name': '3', 'value': 0.3}, {'name': '4', 'value': 0.4}]
pr_key_x = 'threshold'
pr_key_y = 'curve'
lc_key_x = 'epoch'
lc_key_y = 'test_loss'



data = {}
nr_tests = 0
for folder in folders:
    data[folder] = {}
    for sub in sub_folders:
        paths = os.listdir(os.path.join(path, folder, sub['name']))
        nr_tests += len(paths)
        print("Folder", folder, "Sub-folder", sub['name'], "length", len(paths))
        data[folder][sub['name']] = []
        for data_path in paths:
            json_data = util.open_json_result(os.path.join(path, folder, sub['name'], data_path))
            if type(json_data) is list:
                d = json_data[0]
            else:
                d = json_data
            data[folder][sub['name']].append(d)


#Average and find breakeven points, for final series.
breakeven_points = {}
for folder in folders:
    breakeven_points[folder] = []
    for sub in sub_folders:
        pr_avg = util.average(data[folder][sub['name']], pr_key_y, pr_key_x)
        breakeven = util.find_breakeven(pr_avg, samples=4)
        breakeven_points[folder].append({"x": sub['value'], "y": breakeven[1]})
print breakeven_points
#series.append({"name": folder, "data": pr_avg})
#

#Summary figure for MSE loss
loss_points = {}
for folder in folders:
    loss_points[folder] = []
    for sub in sub_folders:
        loss_avg = util.average(data[folder][sub['name']], 'events', lc_key_x)
        last_loss = loss_avg[-1][lc_key_y]
        loss_points[folder].append({"x": sub['value'], "y": last_loss})


pr_series = [{"name": folder, "data": breakeven_points[folder]} for i, folder in enumerate(folders)]
lc_series = [{"name": folder, "data": loss_points[folder]} for folder in folders]
#series.append({"name": folder, "data": loss_avg, "y_key": lc_key_y})
util.display_noise_summary(pr_series, x_label="Label noise %", y_label="precision recall breakeven")
util.display_noise_summary(lc_series, x_label="Label noise %", y_label="MSE loss")