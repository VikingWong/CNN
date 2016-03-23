from util import Params
import secret

#Create secret python file and variable token
token = secret.token
verbose = True
number_of_epochs = 150
dataset_path =  '/home/olav/Pictures/Mass_roads_alpha'
pr_path =       '/home/olav/Pictures/Mass_roads_alpha'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl",
        "curriculum_teacher"    : "./results/curriculum.pkl",
        "curriculum_location"   : "/media/olav/Data storage/dataset/Mass_roads_curriculum_50-baseline"

    })

visual_params = Params({
        "endpoint"              : "http://178.62.232.71/",
        "gui_enabled"           : True
    })

optimization_params = Params({
        "backpropagation"                   : "sgd_nesterov",
        "batch_size"                        : 64,
        "l2_reg"                            : 0.0002,
        "momentum"                          : 0.9,
        "initial_patience"                  : 100000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.997,
        "learning_rate"                     : 0.0011,
        "learning_adjustment"               : 50,
        "learning_decrease"                 : 0.9,
        "factor_rate"                       : 1,
        "factor_adjustment"                 : 100,
        "factor_decrease"                   : 0.998,
        "factor_minimum"                    : 0.8,
        "curriculum_enable"                 : False,
        "curriculum_start"                  : 70,
        "curriculum_adjustment"             : 10
    })
#Reduce, is needed especially for testing and validation. For large samples_per_image, testing validation might not fit on GPU
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "loader"                : "AerialDataset",
    "samples_per_image"     : 50,
    "dataset_std"           : 0.18893923860059578,
    "valid_std"             : 0.19088566314428751,
    "test_std"              : 0.18411163301559019,
    "use_label_noise"       : True,
    "label_noise"           : 0.5,
    "use_rotation"          : True,
    "use_preprocessing"     : True,
    "only_mixed_labels"     : False,
    "mix_ratio"             : 0.5,
    "reduce_training"       : 1,
    "reduce_testing"        : 2,
    "reduce_validation"     : 4,
    "input_dim"             : 64,
    "output_dim"            : 16,
    "chunk_size"            : 2048
})

model_params = Params({
    "loss"              : "crosstrapping",
    "nr_kernels"        : [64, 112, 80 ],
    "random_seed"       : 23455,
    "input_data_dim"    : (3, 64, 64),
    "output_label_dim"  : (16, 16),
    "hidden_layer"      : 4096,
    "dropout_rates"     : [1.0, 0.9, 0.8, 0.5, 1.0],
    "conv_layers"       :
        [
            {"filter": (13,13), "stride": (4, 4), "pool": (2, 2)},
            {"filter": (4, 4), "stride": (1, 1), "pool": (1, 1)},
            {"filter": (3,3), "stride": (1, 1), "pool": (1, 1)}
        ],
})