import sys, os

#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section
from storage import ParamStorage
from model import ConvModel
from aerial import Visualizer

print_section('TOOLS: Visualize result from model')
print("-data: Path to image you want predictions for")
image_path = None
if '-data' in sys.argv:
    idx = sys.argv.index('-d')
    image_path = sys.argv[idx+1]
    print("using {} as image path".format(image_path))

else:
    image_path = '/home/olav/Pictures/Mass_roads/test/data/10378780_15.tiff'

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")

data['model'].hidden_dropout = 0 #Fix
batch_size = data['optimization'].batch_size

v = Visualizer(data['model'], data['params'], data['dataset'])

img = v.visualize(image_path, batch_size, threshold=1)
img.show()
img.save('./tools/visualize/tester.jpg')