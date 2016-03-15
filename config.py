from util import Params
import secret

#Create secret python file and variable token
token = secret.token
verbose = True
number_of_epochs = 600
dataset_path =  '/media/olav/Data storage/dataset/Mass_roads_curriculum_big'
pr_path =       '/home/olav/Pictures/Mass_roads_alpha'
filename_params = Params({
        "results"               : "./results",
        "network_save_name"     : "./results/params.pkl",
        "curriculum_teacher"    : "./results/curriculum.pkl",
        "curriculum_location"   : "/media/olav/Data storage/dataset/Mass_roads_noncurriculum_big"

    })

visual_params = Params({
        "endpoint"              : "http://178.62.232.71/",
        "gui_enabled"           : True
    })

optimization_params = Params({
        "backpropagation"                   : "sgd_nesterov",
        "batch_size"                        : 64,
        "l2_reg"                            : 0.00015,
        "momentum"                          : 0.9,
        "initial_patience"                  : 100000,
        "patience_increase"                 : 2,
        "improvement_threshold"             : 0.997,
        "learning_rate"                     : 0.0014,
        "learning_adjustment"               : 120,
        "learning_decrease"                 : 0.9,
        "factor_rate"                       : 1,
        "factor_adjustment"                 : 200,
        "factor_decrease"                   : 0.998,
        "factor_minimum"                    : 0.8,
        "curriculum_enable"                 : True,
        "curriculum_start"                  : 30,
        "curriculum_adjustment"             : 15
    })
#Reduce, is needed especially for testing and validation. For large samples_per_image, testing validation might not fit on GPU
#Dataset_std can by calculated by dataset_std tool inside tools directory.
dataset_params = Params({
    "loader"                : "AerialCurriculumDataset",
    "samples_per_image"     : 400,
    "dataset_std"           : 0.448638984229,
    "valid_std"             : 0.19088566314428751,
    "test_std"              : 0.18411163301559019,
    ""
    "use_rotation"          : True,
    "use_preprocessing"     : True,
    "only_mixed_labels"     : False,
    "mix_ratio"             : 0.5,
    "reduce_training"       : 1,
    "reduce_testing"        : 0.3,
    "reduce_validation"     : 0.9,
    "input_dim"             : 64,
    "output_dim"            : 16,
    "chunk_size"            : 2048
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
            {"filter": (13,13), "stride": (4, 4), "pool": (2, 2)},
            {"filter": (4, 4), "stride": (1, 1), "pool": (1, 1)},
            {"filter": (3,3), "stride": (1, 1), "pool": (1, 1)}
        ],
})