import sys, os
import numpy as np
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section, print_action
from storage import ParamStorage
from config import filename_params, dataset_params, dataset_path
from dataset_create import CurriculumDataset


print_section("Creating curriculum learning dataset")

# Baseline will create a curriculum with no example ordering, but same amount of examples.
# Avoids results from curriculum learning to be caused by the model just having seen more examples.
is_baseline = False
if '-baseline' in sys.argv:
    is_baseline = True
    print_action("Creating baseline dataset. No curriculum, but same structure")

stages = None
if '-stages' in sys.argv:
    idx = sys.argv.index('-stages')
    stages = np.array(eval(sys.argv[idx+1]))
    print_action("Dataset with threshold-stages of".format(stages))

#Load the curriculum teacher which provide consistency estimates for extracted examples.
store = ParamStorage()
teacher = store.load_params(path=filename_params.curriculum_teacher)

generator = CurriculumDataset(teacher, dataset_path, filename_params.curriculum_location, dataset_params)
generator.create_dataset(is_baseline, thresholds=stages)