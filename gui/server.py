import unirest
import json
from config import model_params, optimization_params, dataset_params, filename_params, visual_params, \
    number_of_epochs, verbose, dataset_path, token
from util import print_error



base_url = visual_params.endpoint
stop = False
test = False #Should only retrieved using is_testing()
default_headers = {
    "Accept": "application/json", "content-type": "application/json", "data-type": "json", "Authorization": token
}
current_id = "none"

def is_testing():
    '''
    If test flag is set via get command status, test contain an epoch number.
    If network is at that number debug functionality is run, and images are generated and displayed.
    '''
    global test
    if test:
        test = False
        return True
    else:
        return False

def get_command_status():
    print(base_url)
    url = base_url + "job/" + current_id + "/status"
    def callback(response):
        print(response.body)
        global stop, test
        if not response.body['running']:
            stop = True
        if response.body['test']:
            test = response.body['test']
    thread = unirest.get(url, headers=default_headers, callback=callback)

def append_job_update( epoch, training_loss, validation_loss, test_loss):
    url = base_url + "job/" +  current_id + "/update"
    data = json.dumps({
        "epoch": epoch,
        "validation_loss": validation_loss,
        "test_loss": test_loss
    })

    def callback(response):
        pass
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
        if(response.body == 'Unauthorized'):
            print_error('Gui is enabled, but token in secret.py is invalid')
            raise Exception('Token is invalid')
        global current_id
        current_id = response.body['id']

    thread = unirest.post(url, headers=default_headers, params=data, callback=callback)

def stop_job(report):
    url = base_url + "job/" + current_id + "/stop"
    print(report)
    thread = unirest.post(url, headers=default_headers, params=json.dumps(report))
