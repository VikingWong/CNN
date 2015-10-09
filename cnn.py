from evaluator import Evaluator
from model import Model
from data import MnistDataset, AerialDataset
from storage.store import ParamStorage
import os
from util import Params
import util

def run_cnn(model_params, optimization_params, dataset, dataset_params, filename_params, visual_params, epochs, verbose=False):
    print(filename_params)
    if not os.path.exists(filename_params.results):
        os.makedirs(filename_params.results)

    d = AerialDataset()
    d.load(dataset, dataset_params) #Input stage
    m = Model(model_params, verbose=True) #Create network stage
    e = Evaluator(m, d)
    e.evaluate(optimization_params, epochs=epochs,  verbose=verbose)

    #Stores the model params. Model can later be restored.
    p = ParamStorage(path=filename_params.network_save_name)
    p.store_params(m.params)

verbose = True
number_of_epochs = 30
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "/params.pkl"

    })
#TODO: Use this for something
visual_params =  Params({
        "visualize_flag"        : False,
    })

optimization_params =  Params({
        "batch_size"                        : 64,
        "initial_learning_rate"             : 0.1,
        "l2_reg"                            : 0.0001,
        "initial_patience"                  : 10000,
        "patiencec_increase"                : 2,
        "improvement_threshold"             : 0.995

    })

#Reduce is for dev purposes. Use a fraction of train dataset
dataset_params = Params({
    "samples_per_image": 20,
    "use_rotation": True,
    "reduce": 0.1
})

model_params =  Params({

    "nr_kernels": [ 64, 112 ],
    "random_seed": 23455,
    "input_data_dim": (3, 64, 64)
     })

run_cnn(
    model_params            = model_params,
    optimization_params     = optimization_params,
    dataset                 = 'C:\\Users\Olav\\Pictures\\Mass_roads',
    dataset_params          = dataset_params,
    filename_params         = filename_params,
    visual_params           = visual_params,
    epochs                  = number_of_epochs,
    verbose                 = verbose,
)