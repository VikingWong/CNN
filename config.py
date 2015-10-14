from util import Params

verbose = True
number_of_epochs = 5
dataset_path = 'C:\\Users\\olav\\Pictures\\Mass_roads_overfitting_test'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl"

    })
#TODO: Use this for something
visual_params =  Params({
        "visualize_flag"        : False,
    })

optimization_params =  Params({
        "batch_size"                        : 16,
        "initial_learning_rate"             : 0.009,
        "l2_reg"                            : 0.0001,
        "initial_patience"                  : 10000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.995

    })

#Reduce is for dev purposes. Use a fraction of train dataset
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "samples_per_image"     : 200,
    "dataset_std"           : 0.233174571944,
    "use_rotation"          : True,
    "reduce"                : 1,
    "input_dim"             : 64,
    "output_dim"            : 16
})

model_params =  Params({

    "nr_kernels"            : [ 64, 112 ],
    "random_seed"           : 23455,
    "input_data_dim"            : (3, 64, 64)
     })