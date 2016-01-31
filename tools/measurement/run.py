import sys, os
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section
from model import ConvModel
from storage import ParamStorage
from precisionrecall import PrecisionRecallCurve

print_section("TOOLS: Measure precision and recall of model")

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")
m = ConvModel(data['model'])
dataset_path = '/home/olav/Pictures/Mass_roads' #TODO: default, but specifiy as environment variable
batch_size = data['optimization'].batch_size
measurer = PrecisionRecallCurve(dataset_path, m, data['params'], data['model'], data['dataset'])
datapoints = measurer.get_curves_datapoints(batch_size)