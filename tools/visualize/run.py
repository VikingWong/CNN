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
from gui.server import send_result_images

print_section('TOOLS: Visualize result from model')
print("-data: Path to image you want predictions for")
image_path = None
if '-data' in sys.argv:
    idx = sys.argv.index('-data')
    image_path = sys.argv[idx+1]
    print_action("using {} as image path".format(image_path))

else:
    image_path = '/home/olav/Pictures/Mass_roads/test/data/10378780_15.tiff'

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

data['model'].hidden_dropout = 0 #Fix
batch_size = data['optimization'].batch_size

v = Visualizer(data['model'], data['params'], data['dataset'])
image_prediction, image_hit = v.visualize(image_path, batch_size, threshold=1)

out = Image.resize(image_prediction, 0.5)
out2 =  Image.resize(image_hit, 0.5)

if store_gui:
    buf= StringIO.StringIO()
    buf2= StringIO.StringIO()
    out.save(buf, format='JPEG')
    out2.save(buf2, format='JPEG')
    send_result_images(job_id, buf.getvalue(), buf2.getvalue())

out.save('./tools/visualize/pred.jpg')
out2.save('./tools/visualize/hit.jpg')

out.show()
out2.show()