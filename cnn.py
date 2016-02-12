from evaluator import Evaluator
from model import ConvModel
from data import AerialDataset
from storage.store import ParamStorage
import gui.server
import os
from printing import print_section
from config import model_params, optimization_params, dataset_params, filename_params, visual_params, \
    number_of_epochs, verbose, dataset_path
from tools.measurement import PrecisionRecallCurve

def run_cnn(model_params, optimization_params, dataset_path, dataset_params, filename_params, visual_params, epochs, verbose=False):
    print(filename_params)
    if not os.path.exists(filename_params.results):
        os.makedirs(filename_params.results)

    dataset = AerialDataset()
    dataset.load(dataset_path, dataset_params, optimization_params.batch_size) #Input stage
    model = ConvModel(model_params, verbose=True) #Create network stage
    evaluator = Evaluator(model, dataset, optimization_params)
    evaluator.run(epochs=epochs,  verbose=verbose)
    report = evaluator.get_result()
    dataset.destroy()

    print_section('Evaluation precision and recall')
    #TODO: Test destroy
    prc = PrecisionRecallCurve(dataset_path, model.params, model_params, dataset_params)
    datapoints = prc.get_curves_datapoints(optimization_params.batch_size)
    #Stores the model params. Model can later be restored.
    print_section('Storing model parameters')
    storage = ParamStorage(path=filename_params.network_save_name)
    storage.store_params(model.params, model_params, dataset_params, optimization_params, number_of_epochs)
    if visual_params.gui_enabled:
        gui.server.send_precision_recall_data(datapoints)
        gui.server.stop_job(report)


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