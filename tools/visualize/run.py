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
from interface.command import get_command

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
is_image_path, image_path = get_command('-data', default='/home/olav/Pictures/Mass_roads/test/data/10378780_15.tiff')

store_data_image, = get_command('-storeimage')

store_gui, job_id = get_command('-store', default="None")

is_tradeoff, bto = get_command('-tradeoff', default="0.5")
store = ParamStorage()
data = store.load_params(path="./results/params.pkl")

batch_size = data['optimization'].batch_size

v = Visualizer(data['model'], data['params'], data['dataset'])
image_prediction, image_hit, image_data = v.visualize(image_path, batch_size, best_trade_off=bto)

store_image(image_prediction, job_id, store_gui)
store_image(image_hit, job_id, store_gui)

if store_data_image:
    store_image(image_data, job_id, store_gui)
