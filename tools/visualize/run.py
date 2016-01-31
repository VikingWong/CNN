import sys, os
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from storage import ParamStorage
from model import ConvModel
from aerial import Visualizer

print(sys.path)
store = ParamStorage()
data = store.load_params(path="./results/params.pkl")
print(data)
m = ConvModel(data['model'])
dataset_std = data['dataset'].dataset_std

v = Visualizer(m, data['params'], std=dataset_std)
img = v.visualize()
img.show()
img.save('./tools/visualize/tester.jpg')