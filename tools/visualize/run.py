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

'''
This tool creates a model from saved params, and stitch together predictions, to qualitatively show performance of model.
There are also options to upload image to GUI. The best tradeoff between precision and recall should be specified as
a parameter and the actual aerial image is set by supplying the path in console by the -data property.

The tool creates, and saves the prediction stitch image, as well as a hit and miss image. This image show, where the
prediction are correct (green), where they are missing (red) and where they should not be according to the label (blue).
'''

def store_image(image, job_id, store_gui, name="image"):
    out = Image.resize(image, 0.5)

    if store_gui:
        buf= StringIO.StringIO()
        out.save(buf, format='JPEG')
        send_result_image(job_id, buf.getvalue())

    image.save('./tools/visualize/'+ name +'.jpg')
    image.show()

print_section('TOOLS: Visualize result from model')
print("-data: Path to image in dataset you want visualization of | -store_gui: Upload images to exp with supplied id | \
      -tradeoff: Threshold value associated with precision recall breakeven |-storeimage: Include aerial image")

is_image_path, image_path = get_command('-data', default='/home/olav/Pictures/Mass_roads/test/data/10378780_15.tiff')

store_data_image, temp = get_command('-storeimage')

store_gui, job_id = get_command('-store_gui', default="None")

is_tradeoff, bto = get_command('-tradeoff', default="0.5")
bto = float(bto)

is_model, model_path = get_command('-model', default="./results/params.pkl")
store = ParamStorage()
data = store.load_params(path=model_path)

batch_size = data['optimization'].batch_size

v = Visualizer(data['model'], data['params'], data['dataset'])
image_prediction, image_hit, image_data = v.visualize(image_path, batch_size, best_trade_off=bto)

store_image(image_prediction, job_id, store_gui, name="pred")
store_image(image_hit, job_id, store_gui, name="hit")

if store_data_image:
    store_image(image_data, job_id, store_gui, name="image")
