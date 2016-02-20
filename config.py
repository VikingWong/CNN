from util import Params
import secret

token = secret.token
verbose = True
number_of_epochs = 300
dataset_path = '/home/olav/Pictures/Mass_roads'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl"

    })

visual_params = Params({
        "endpoint"              : "http://178.62.232.71/",
        "gui_enabled"           : True
    })

optimization_params = Params({
        "batch_size"                        : 64,
        "l2_reg"                            : 0.0001,
        "momentum"                          : 0.9,
        "initial_patience"                  : 100000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.997,
        "backpropagation"                   : "sgd_nesterov",
        "learning_rate"                     : 0.0009,
        "learning_adjustment"               : 30,
        "learning_decrease"                 : 0.95,
        "factor_rate"                       : 1,
        "factor_adjustment"                 : 100,
        "factor_decrease"                   : 0.998,
        "factor_minimum"                    : 0.8
    })

#Reduce is for dev purposes. Use a fraction of train dataset
#Dataset_std can by calculated by dataset_std tool inside tools directory.
#TODO: last chunk so small so training loss is misleading
dataset_params = Params({
    "samples_per_image"     : 400,
    "dataset_std"           : 0.233174571944,
    "use_rotation"          : True,
    "use_preprocessing"     : True,
    "only_mixed_labels"     : True,
    "mix_ratio"             : 0.5,
    "reduce_training"       : 1,
    "reduce_testing"        : 0.3,
    "input_dim"             : 64,
    "output_dim"            : 16,
    "chunk_size"            : 2048
})

model_params = Params({
    "loss"              : "bootstrapping",
    "nr_kernels"        : [64, 112, 80 ],
    "random_seed"       : 23455,
    "input_data_dim"    : (3, 64, 64),
    "output_label_dim"  : (16, 16),
    "hidden_layer"      : 4096,
    "dropout_rates"     : [0.9, 0.85, 0.75, 0.5, 1.0],
    "conv_layers"       :
        [
            {"filter": (16,16), "stride": (4, 4), "pool": (2, 2)},
            {"filter": (4, 4), "stride": (1, 1), "pool": (1, 1)},
            {"filter": (3,3), "stride": (1, 1), "pool": (1, 1)}
        ],
})