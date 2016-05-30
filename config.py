from util import Params
import secret

'''
This file contains the configurable hyperparameters of the CNN. For instance the backprogagation method,
the number of convolutional layers and their architecutre, curriculum learning behavior etc.
'''
#Create secret python file and variable token
token = secret.token
verbose = True
number_of_epochs = 100
dataset_path =  '/home/olav/Pictures/Mass_roads_alpha'
pr_path =       '/home/olav/Pictures/Mass_roads_alpha'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl",
        "curriculum_teacher"    : "/home/olav/Documents/Results/E7_inexperienced_teacher/teacher/params.pkl",
        "curriculum_location"   : "/media/olav/Data storage/dataset/Mass_inexperienced_100-2-5-stages"

    })

visual_params = Params({
        "endpoint"              : "http://178.62.232.71/",
        "gui_enabled"           : True
    })

optimization_params = Params({
        "backpropagation"                   : "sgd_nesterov",
        "batch_size"                        : 64,
        "l2_reg"                            : 0.0001,
        "momentum"                          : 0.93,
        "initial_patience"                  : 500000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.997,

        "learning_rate"                     : 0.0025,
        "learning_adjustment"               : 15,
        "learning_decrease"                 : 0.90,

        "factor_rate"                       : 1.0,
        "factor_adjustment"                 : 60,
        "factor_decrease"                   : 0.95,
        "factor_minimum"                    : 0.8,

        "curriculum_enable"                 : False,
        "curriculum_start"                  : 30,
        "curriculum_adjustment"             : 15
    })
#Reduce, is needed especially for testing and validation. For large samples_per_image, testing validation might not fit on GPU
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "loader"                : "AerialDataset",
    "with_replacement"      : True, #True: random indexes in training set replaced 62% average. False: Entire training set replaced.
    "samples_per_image"     : 200,
    #"dataset_std"           : 0.18945282966287444, #Norwegian dataset
    "dataset_std"           : 0.18893923860059578, #Mass
    "valid_std"             : 0.19088566314428751, #Not used
    "test_std"              : 0.18411163301559019, #Not used
    "reduce_training"       : 1.0,
    "reduce_testing"        : 0.5,
    "reduce_validation"     : 1.0,
    "use_rotation"          : True,
    "use_preprocessing"     : True,
    "input_dim"             : 64,
    "output_dim"            : 16,
    "chunk_size"            : 1024,

    "use_label_noise"       : True,
    "label_noise"           : 0.0,

    "only_mixed_labels"     : True,
    "mix_ratio"             : 0.5
})

model_params = Params({
    "loss"              : "bootstrapping",
    "nr_kernels"        : [64, 112, 80 ],
    "random_seed"       : 23455,
    "input_data_dim"    : (3, 64, 64),
    "output_label_dim"  : (16, 16),
    "hidden_layer"      : 4096,
    "dropout_rates"     : [0.85, 0.85, 0.85, 0.5, 1.0],
    "conv_layers"       :
        [
            {"filter": (16,16), "stride": (2, 2), "pool": (2, 2)},
            {"filter": (8, 8), "stride": (1, 1), "pool": (1, 1)},
            {"filter": (3,3), "stride": (1, 1), "pool": (1, 1)}
        ],
})