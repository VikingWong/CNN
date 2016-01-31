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
m = ConvModel(data.model)
dataset_std = data.dataset.dataset_std

measurer = PrecisionRecallCurve(m, data.params, std=dataset_std)
datapoints = measurer.get_curves_datapoints()