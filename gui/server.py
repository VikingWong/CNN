import unirest
import json
from config import model_params, optimization_params, dataset_params, filename_params, visual_params, \
    number_of_epochs, verbose, dataset_path



base_url = visual_params.endpoint
stop = False
default_headers = { "Accept": "application/json", "content-type": "application/json", "data-type": "json" }
current_id = "none"


def get_stop_status():
    print(base_url)
    url = base_url + "job/" + current_id + "/status"
    def callback(response):
        global stop
        print(response.body['msg'])
        if not response.body['running']:
            stop = True
    thread = unirest.get(url, headers=default_headers, callback=callback)

def append_job_update( epoch, training_loss, validation_loss, test_loss):
    url = base_url + "job/" +  current_id + "/update"
    data = json.dumps({
        "epoch": epoch,
        "validation_loss": validation_loss,
        "test_loss": test_loss
    })

    def callback(response):
        print(response.code)
    thread = unirest.post(url, headers=default_headers, callback=callback, params=data)

def start_new_job():

    url = base_url + "job/start/"
    data = {
        "model_params": model_params.__dict__,
        "optimization_params": optimization_params.__dict__,
        "dataset_params": dataset_params.__dict__,
        "filename_params": filename_params.__dict__,
        "epochs": number_of_epochs,
        "dataset_path": dataset_path
    }
    data = json.dumps(data)

    def callback(response):
        global current_id
        current_id = response.body['id']

    thread = unirest.post(url, headers=default_headers, params=data, callback=callback)

def stop_job():
    url = base_url + "job/" + current_id + "/stop"
    thread = unirest.post(url, headers=default_headers)