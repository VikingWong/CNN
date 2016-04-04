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

print_section("TOOLS: Measure precision and recall of model")
print("-data: path to dataset | -store: job id to store curve in GUI")

is_dataset_path, dataset_path = get_command('-data', default='/home/olav/Pictures/Mass_roads_alpha')

store_gui,
store_gui = False
job_id = "-1"
if '-store_gui' in sys.argv:
    idx = sys.argv.index('-store_gui')

    if len(sys.argv) > idx+1:
        store_gui = True
        job_id = sys.argv[idx+1]
        print_action("Storing precision recall curve in database for job {}".format(job_id))

store_path = "./pr_data.json"
if '-store_path' in sys.argv:
    idx = sys.argv.index('-store_path')
    if len(sys.argv) > idx+1:
        store_path = sys.argv[idx+1]
        print_action("Storing precision recall curve as file")

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")
batch_size = data['optimization'].batch_size

measurer = PrecisionRecallCurve(dataset_path, data['params'], data['model'], data['dataset'])
datapoints = measurer.get_curves_datapoints(batch_size)
#datapoints.sort(key=lambda p: p['recall'])

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