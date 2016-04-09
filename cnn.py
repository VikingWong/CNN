import os, sys
import StringIO

from evaluator import Evaluator
from model import ConvModel
from data import DataLoader
from storage.store import ParamStorage
import interface
import printing
from config import model_params, optimization_params, dataset_params, filename_params, visual_params, \
    number_of_epochs, verbose, dataset_path, pr_path
from tools.measurement import PrecisionRecallCurve

def run_cnn(model_params, optimization_params, dataset_path, dataset_params, filename_params, visual_params, epochs, verbose=False):
    print(filename_params)
    if not os.path.exists(filename_params.results):
        os.makedirs(filename_params.results)

    is_config, config_values = interface.command.get_command("-config")
    is_curriculum, curriculum_set = interface.command.get_command("-curriculum")
    is_batch_run, batch_index = interface.command.get_command("-batch", default="0")
    is_init_params, param_path = interface.command.get_command("-params")

    if is_config:
        #Assume  config is speficially for running bootstrapping batches.
        config_arr = eval(config_values)
        if len(config_arr) == 2:
            loss_function = config_arr[0]
            label_noise = float(config_arr[1])
            dataset_params.label_noise = label_noise
            model_params.loss = loss_function
            batch_index = loss_function + "-" + str(label_noise) + "-" + batch_index
            print(batch_index)

    if is_curriculum:
        dataset_path = curriculum_set

    weights = None
    if is_init_params:
        store = ParamStorage()
        if not param_path:
            param_path = "./results/params.pkl"
        weights = store.load_params(path=param_path)['params']


    dataset = DataLoader.create()
    dataset.load(dataset_path, dataset_params, optimization_params.batch_size) #Input stage
    model = ConvModel(model_params, verbose=True) #Create network stage
    #TODO: FIX: Correct datasetpath not saved propertly
    evaluator = Evaluator(model, dataset, optimization_params, dataset_path)
    evaluator.run(epochs=epochs,  verbose=verbose, init=weights)
    report = evaluator.get_result()

    network_store_path = filename_params.network_save_name
    result_path = filename_params.results + "/results.json"
    if is_batch_run:
        network_store_path = filename_params.results + "/batch" + batch_index +  ".pkl"
        result_path =filename_params.results + "/batch" + batch_index +  ".json"

    storage = ParamStorage(path=network_store_path)
    storage.store_params(model.params)

    dataset.destroy()

    if visual_params.gui_enabled:
         interface.server.stop_job(report)

    printing.print_section('Evaluation precision and recall')

    prc = PrecisionRecallCurve(pr_path, model.params, model_params, dataset_params)
    test_datapoints = prc.get_curves_datapoints(optimization_params.batch_size, set_name="test")
    valid_datapoints = prc.get_curves_datapoints(optimization_params.batch_size, set_name="valid")
    #Stores the model params. Model can later be restored.
    printing.print_section('Storing model parameters')

    if visual_params.gui_enabled:
        interface.server.send_precision_recall_data(test_datapoints, valid_datapoints)
    storage.store_result(result_path, evaluator.events, test_datapoints, valid_datapoints)


run_cnn(
    model_params            = model_params,
    optimization_params     = optimization_params,
    dataset_path            = dataset_path,
    dataset_params          = dataset_params,
    filename_params         = filename_params,
    visual_params           = visual_params,
    epochs                  = number_of_epochs,
    verbose                 = verbose,
)