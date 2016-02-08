import sys, os
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section, print_action

from storage import ParamStorage
from precisionrecall import PrecisionRecallCurve
from gui.server import send_precision_recall_data

print_section("TOOLS: Measure precision and recall of model")
print("-data: path to dataset | -store: job id to store curve in GUI")

dataset_path = None
if '-data' in sys.argv:
    idx = sys.argv.index('-data')
    dataset_path = sys.argv[idx+1]
    print_action("using {} as image path".format(dataset_path))
else:
    dataset_path = '/home/olav/Pictures/Mass_roads'


store_gui = False
job_id = "-1"
if '-store' in sys.argv:
    idx = sys.argv.index('-store')

    if len(sys.argv) > idx+1:
        store_gui = True
        job_id = sys.argv[idx+1]
        print_action("Storing precision recall curve in database for job {}".format(job_id))

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")
batch_size = data['optimization'].batch_size

measurer = PrecisionRecallCurve(dataset_path, data['params'], data['model'], data['dataset'])
datapoints = measurer.get_curves_datapoints(batch_size)

if store_gui:
    send_precision_recall_data(datapoints, job_id=job_id)