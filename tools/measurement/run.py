import sys, os
import matplotlib.pyplot as plt
import json

#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section, print_action
from storage import ParamStorage
from precisionrecall import PrecisionRecallCurve
from interface.server import send_precision_recall_data
from interface.command import get_command

'''
This tool creates the datapoints necessary for a precision and recall curve figure. The tool samples a patch dataset
from the test and validation set, and creates predictions using a trained model (-model). These predictions are thresholded
at several values. The binarized predictions and the label are then used to calculate the precision as well as the recall.
These values including the threshold amount constitute a data point. Supplying a experiment id (-store_gui), will
store the datapoints in the web GUI.

It's worth noting that the measurements are relaxed. Relaxed precision and relaxed recall. This is implemented by the
image processing operation, dilation. The slack variable is set to 3 pixels.
'''

print_section("TOOLS: Measure precision and recall of model")
print("-data: path to dataset | -store: job_gui id to store curve in GUI | -store_path: store results locally")

#====== Arguments ===============================================
is_dataset_path, dataset_path = get_command('-data', default='/home/olav/Pictures/Mass_roads_alpha')
store_gui, job_id = get_command('-store_gui', default='-1')
is_store_path, store_path = get_command('-store_path', default='./pr_data.json')
is_model, model_path = get_command('-model', default='./results/params.pkl')
#==============================================================

store = ParamStorage()
data = store.load_params(path=model_path)
batch_size = data['optimization'].batch_size

measurer = PrecisionRecallCurve(dataset_path, data['params'], data['model'], data['dataset'])
datapoints = measurer.get_curves_datapoints(batch_size)

if store_gui:
    send_precision_recall_data(datapoints, None, job_id=job_id)
else:
    with open(store_path, 'w') as outfile:
        json.dump([{"curve": datapoints, "events": []}], outfile)


plt.suptitle('Precision and recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.plot([p['recall'] for p in datapoints], [p['precision'] for p in datapoints])
plt.show(p)