import sys, os
import numpy as np
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from interface.command import get_command
from printing import print_section, print_action
from storage import ParamStorage
from config import filename_params, dataset_params, dataset_path
from dataset_create import CurriculumDataset


print_section("Creating curriculum learning dataset")

# Baseline will create a curriculum with no example ordering, but same amount of examples.
# Avoids results from curriculum learning to be caused by the model just having seen more examples.
is_baseline, baseline = get_command('-baseline')

is_stages, stages = get_command('-stages', default="[0.1, 1.0]")
stages = np.array(eval(stages))

#Precision recall breakeven point.
is_tradeoff, tradeoff = get_command('-tradeoff')
if is_tradeoff:
    tradeoff = float(tradeoff)
    
#Load the curriculum teacher which provide consistency estimates for extracted examples.
store = ParamStorage()
teacher = store.load_params(path=filename_params.curriculum_teacher)

generator = CurriculumDataset(teacher, dataset_path, filename_params.curriculum_location, dataset_params, tradeoff)
generator.create_dataset(is_baseline, thresholds=stages)