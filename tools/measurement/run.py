import sys, os
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from model import ConvModel
from storage import ParamStorage
from precisionrecall import PrecisionRecallCurve

#TODO: Will placement in init cause conflict?
store = ParamStorage()
data = store.load_params(path="./results/params.pkl")
print(data)
m = ConvModel(data['model'])
dataset_std = data['dataset'].dataset_std

measurer = PrecisionRecallCurve(m, data['params'], std=dataset_std)
datapoints = measurer.get_curves_datapoints()