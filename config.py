from util import Params
import secret

token = secret.token
verbose = True
number_of_epochs = 1
dataset_path = '/home/olav/Pictures/Mass_roads'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl"

    })
#TODO: Use this for something
visual_params = Params({
        "endpoint"              : "http://178.62.232.71/",
        "gui_enabled"           : True
    })

#TODO: Implement dropout_rate
optimization_params = Params({
        "batch_size"                        : 32,
        "initial_learning_rate"             : 0.095,
        "epoch_learning_adjustment"         : 5,
        "learning_rate_decrease"            : 0.95,
        "l2_reg"                            : 0.0004,
        "momentum"                          : 0.9,
        "initial_patience"                  : 100000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.995,
        "backpropagation"                   : "sgd_nesterov"
    })

#Reduce is for dev purposes. Use a fraction of train dataset
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "samples_per_image"     : 40,
    "dataset_std"           : 0.233174571944,
    "use_rotation"          : True,
    "use_preprocessing"     : True,
    "only_mixed_labels"     : False,
    "mix_ratio"             : 0.5,
    "reduce"                : 0.01,
    "input_dim"             : 64,
    "output_dim"            : 16,
    "chunk_size"            : 2048
})

#TODO: BIg problem using stride or subsample. Should simply not use it, since gpu is not utilized
model_params = Params({

    "nr_kernels"        : [64, 112, 80 ],
    "random_seed"       : 23455,
    "input_data_dim"    : (3, 64, 64),
    "output_label_dim"  : (16, 16),
    "hidden_layer"      : 4096,
    "hidden_dropout"    : 0.5,
    "conv_layers"       :
        [
            {"filter": (16,16), "stride": (1, 1), "pool": (4, 4)},
            {"filter": (4, 4), "stride": (1, 1), "pool": (1, 1)},
            {"filter": (3,3), "stride": (1, 1), "pool": (1, 1)}
        ],
})