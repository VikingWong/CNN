import sys, os

#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section
from storage import ParamStorage
from model import ConvModel
from aerial import Visualizer

print_section('TOOLS: Visualize result from model')

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")

m = ConvModel(data.model)
dataset_std = data.dataset.dataset_std

v = Visualizer(m, data.params, std=dataset_std)
img = v.visualize()
img.show()
img.save('./tools/visualize/tester.jpg')