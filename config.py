from util import Params

verbose = True
number_of_epochs = 20
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl"

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
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.995

    })

#Reduce is for dev purposes. Use a fraction of train dataset
dataset_params = Params({
    "samples_per_image"     : 5,
    "use_rotation"          : True,
    "reduce"                : 0.1,
    "input_dim"             : 64,
    "output_dim"            : 16
})

model_params =  Params({

    "nr_kernels"            : [ 64, 112 ],
    "random_seed"           : 23455,
    "input_data_dim"            : (3, 64, 64)
     })