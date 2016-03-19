import numpy as np
import sys, os
import json

sys.path.append(os.path.abspath("./"))

import util

#TODO: Mission to create averaging tool. Takes Serveral jsons, align values, and average them. Display them in plot.
#TODO: Load json files
#TODO: Define properties used to extract and match values
#TODO: Numpy to average values into a single series.
#TODO: Display in matplotlib with nice graphics.

path = '/home/olav/Documents/Results/curr100/'
folders = ['baseline', 'curriculum']
pr_key_x = 'threshold'
lc_key_x = 'epoch'
lc_key_y = 'validation'


def average(series, series_key, x_align_key):
    #Assume that all series, and datapoints contain the same keys. Everyting is recorded.
    if len(series) <= 0:
        return []
    nr_datapoints = len(series[0][series_key])
    if nr_datapoints <= 0:
        return []

    keys = series[0][series_key][0].keys()

    #TODO: better way to avoid not summable keys? Check type of each value
    if'date_recorded' in keys:
        d = keys.index('date_recorded')
        del keys[d]

    if'training_rate' in keys:
        d = keys.index('training_rate')
        del keys[d]

    combined = []
    for i in range(nr_datapoints):
        combined.append({})

    for k in keys:
        for j in range(nr_datapoints):
            values = []
            for s in range(len(series)):
                values.append(series[s][series_key][j][k])
            #print(k)
            #print(sum(values)/len(values))
            combined[j][k] = sum(values)/len(values)
    return combined


def open_json_result(file_path):
    data = {}
    with open(file_path) as data_file:
        data = json.load(data_file)
    return data


baseline_paths = os.listdir(os.path.join(path, folders[0]))
test_paths = os.listdir(os.path.join(path, folders[1]))
all_tests = [baseline_paths, test_paths]

data = {folders[0]: [], folders[1]: []}

for t in range(len(all_tests)):
    for data_path in all_tests[t]:
        json_data = open_json_result(os.path.join(path, folders[t],data_path))
        data[folders[t]].append(json_data[0])


pr_avg_baseline = average(data[folders[0]], 'curve', pr_key_x)
pr_avg_test = average(data[folders[1]], 'curve', pr_key_x)
series = [{"name": folders[0], "data": pr_avg_baseline}, {"name": folders[1], "data": pr_avg_test}]
util.display_precision_recall_plot(series)

loss_avg_data = average(data[folders[0]], 'events', lc_key_x)
