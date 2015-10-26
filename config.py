from util import Params

verbose = True
number_of_epochs = 500
dataset_path = '/home/olav/Pictures/mass_test'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl"

    })
#TODO: Use this for something
visual_params =  Params({
        "visualize_flag"        : False,
    })

#TODO: Implement dropout_rate
optimization_params =  Params({
        "batch_size"                        : 2,
        "initial_learning_rate"             : 0.001,
        "l2_reg"                            : 0.001,
        "initial_patience"                  : 10000000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.995

    })

#Reduce is for dev purposes. Use a fraction of train dataset
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "samples_per_image"     : 1024,
    "dataset_std"           : 0.233174571944,
    "use_rotation"          : False,
    "use_preprocessing"     : False,
    "only_mixed_labels"     : True,
    "mix_ratio"             : 0.5,
    "reduce"                : 1,
    "input_dim"             : 64,
    "output_dim"            : 16
})

#TODO: BIg problem using stride or subsample. Should simply not use it, since gpu is not utilized
model_params =  Params({

    "nr_kernels"        : [ 64, 112, 80 ],
    "random_seed"       : 23455,
    "input_data_dim"    : (3, 64, 64),
    "output_label_dim"  : (16,16),
    "hidden_layer"      : 4096,
    "hidden_dropout"    : 0.5,
    "conv_layers"       :
        [
            {"filter": (13,13), "stride": (1, 1), "pool": (4,4)},
            {"filter": (6, 6), "stride": (1, 1), "pool": (2,2)},
            {"filter": (4,4), "stride": (1, 1), "pool": (1,1)}
        ],
})