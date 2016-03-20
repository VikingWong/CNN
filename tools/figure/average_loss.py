import sys, os

sys.path.append(os.path.abspath("./"))

import util

path = '/home/olav/Documents/Results/curr100/'
folder = 'curriculum'
lc_key_x = 'epoch'

print("Creating averaged loss figures")
paths = os.listdir(os.path.join(path, folder))

data = []

for data_path in paths:
    json_data = util.open_json_result(os.path.join(path, folder,data_path))
    data.append(json_data[0])


loss_avg = util.average(data, 'events', lc_key_x)
series = [
    {"name": "Training loss", "data": loss_avg, "y_key":  "training_loss"},
    {"name": "Validation loss", "data": loss_avg, "y_key":  "validation_loss"},
    {"name": "Test loss", "data": loss_avg, "y_key":  "test_loss"}
]
util.display_loss_curve_plot(series)