import sys, os
import StringIO

#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section
from storage import ParamStorage
from model import ConvModel
from aerial import Visualizer
from printing import print_action
import tools.util as Image
from interface.server import send_result_image

def store_image(image, job_id, store_gui):
    out = Image.resize(image, 0.5)

    if store_gui:
        buf= StringIO.StringIO()
        out.save(buf, format='JPEG')
        send_result_image(job_id, buf.getvalue())

    out.save('./tools/visualize/pred.jpg')
    out.show()

print_section('TOOLS: Visualize result from model')
print("-data: Path to image you want predictions for")
image_path = None
if '-data' in sys.argv:
    idx = sys.argv.index('-data')
    image_path = sys.argv[idx+1]
    print_action("using {} as image path".format(image_path))

else:
    image_path = '/home/olav/Pictures/Mass_roads/test/data/10378780_15.tiff'

store_data_image = False
if '-storeimage' in sys.argv:
    print_action("Store data image")
    store_data_image = True



store_gui = False
job_id = "-1"
if '-store' in sys.argv:
    idx = sys.argv.index('-store')

    if len(sys.argv) > idx+1:
        store_gui = True
        job_id = sys.argv[idx+1]
        print_action("Storing images in GUI for job {}".format(job_id))

store = ParamStorage()
data = store.load_params(path="./results/params.pkl")

batch_size = data['optimization'].batch_size

v = Visualizer(data['model'], data['params'], data['dataset'])
bto =0.2201
image_prediction, image_hit, image_data = v.visualize(image_path, batch_size, best_trade_off=bto)

store_image(image_prediction, job_id, store_gui)
store_image(image_hit, job_id, store_gui)

if store_data_image:
    store_image(image_data, job_id, store_gui)
