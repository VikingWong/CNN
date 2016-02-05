import sys, os
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section

from storage import ParamStorage
from precisionrecall import PrecisionRecallCurve
from gui.server import send_precision_recall_data

print_section("TOOLS: Measure precision and recall of model")

store_gui = True

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")
dataset_path = '/home/olav/Pictures/Mass_roads' #TODO: default, but specifiy as environment variable
batch_size = data['optimization'].batch_size
measurer = PrecisionRecallCurve(dataset_path, data['params'], data['model'], data['dataset'])
datapoints = measurer.get_curves_datapoints(batch_size)

if store_gui:
    job_id = "56b1fa7791a7324608856b20"
    send_precision_recall_data(datapoints, job_id=job_id)