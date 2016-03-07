import sys, os
#Makes sh scripts find modules.
sys.path.append(os.path.abspath("./"))

from printing import print_section, print_action
from storage import ParamStorage
from config import filename_params, optimization_params, dataset_params, dataset_path
from dataset_create import CurriculumDataset

#TODO: Dataset_create utilize augmenter for training dataset, also valid and testing. New data loader.
#TODO: Store all samples, store 100 in each file.
#TODO: Estimate sample
#TODO: Load dataset on
#TODO: Must be prepared for each dataset taking in the order of 60gb.
#TODO: Read file-list, shuffle it, so each run do not have exactly the same order.
#TODO: File structure is : curriculum --> main , 1, 2, 3, 4, 5, 6, 7, 8, 9. Main have the most , while 1, 2, 3 contains 10% or something of the main size.

print_section("Creating curriculum learning dataset")

# Baseline will create a curriculum with no example ordering, but same amount of examples.
# Avoids results from curriculum learning to be caused by the model just having seen more examples.
is_baseline = False
if '-baseline' in sys.argv:
    is_baseline = True
    print_action("Creating baseline dataset. No curriculum, but same structure")

#Load the curriculum teacher which provide consistency estimates for extracted examples.
store = ParamStorage()
teacher = store.load_params(path=filename_params.curriculum_teacher)

generator = CurriculumDataset(teacher, dataset_path, filename_params.curriculum_location, dataset_params)
generator.create_dataset()