from util import Params
import secret

#Create secret python file and variable token
token = secret.token
verbose = True
number_of_epochs = 300
dataset_path =  '/media/olav/Data storage/dataset/E6-Mass_3600_performance-baseline'
pr_path =       '/home/olav/Pictures/Mass_roads_alpha'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl",
        "curriculum_teacher"    : "./results/5707c56c9d2760770f4937d9/curriculum.pkl",
        "curriculum_location"   : "/media/olav/Data storage/dataset/Mass_roads_anticurriculum_400_3stage-baseline"

    })

visual_params = Params({
        "endpoint"              : "http://178.62.232.71/",
        "gui_enabled"           : True
    })

optimization_params = Params({
        "backpropagation"                   : "sgd_nesterov",
        "batch_size"                        : 128,
        "l2_reg"                            : 0.0001,
        "momentum"                          : 0.91,
        "initial_patience"                  : 400000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.997,

        "learning_rate"                     : 0.0018,
        "learning_adjustment"               : 50,
        "learning_decrease"                 : 0.9,

        "factor_rate"                       : 1.0,
        "factor_adjustment"                 : 100,
        "factor_decrease"                   : 0.90,
        "factor_minimum"                    : 0.70,

        "curriculum_enable"                 : True,
        "curriculum_start"                  : 60,
        "curriculum_adjustment"             : 30
    })
#Reduce, is needed especially for testing and validation. For large samples_per_image, testing validation might not fit on GPU
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "loader"                : "AerialCurriculumDataset",
    "with_replacement"      : True, #True: random indexes in training set replaced 62% average. False: Entire training set replaced.
    "samples_per_image"     : 400,
    #"dataset_std"           : 0.18945282966287444, #Norwegian dataset
    "dataset_std"           : 0.18893923860059578, #Mass
    "valid_std"             : 0.19088566314428751, #Not used
    "test_std"              : 0.18411163301559019, #Not used
    "reduce_training"       : 1.0,
    "reduce_testing"        : 0.2,
    "reduce_validation"     : 0.4,
    "use_rotation"          : True,
    "use_preprocessing"     : True,
    "input_dim"             : 64,
    "output_dim"            : 16,
    "chunk_size"            : 1024,

    "use_label_noise"       : False,
    "label_noise"           : 0.0,

    "only_mixed_labels"     : True,
    "mix_ratio"             : 0.5
})

model_params = Params({
    "loss"              : "crossentropy",
    "nr_kernels"        : [64, 112, 80 ],
    "random_seed"       : 23455,
    "input_data_dim"    : (3, 64, 64),
    "output_label_dim"  : (16, 16),
    "hidden_layer"      : 4096,
    "dropout_rates"     : [1.0, 0.9, 0.8, 0.5, 1.0],
    "conv_layers"       :
        [
            {"filter": (16,16), "stride": (4, 4), "pool": (2, 2)},
            {"filter": (4, 4), "stride": (1, 1), "pool": (1, 1)},
            {"filter": (3,3), "stride": (1, 1), "pool": (1, 1)}
        ],
})