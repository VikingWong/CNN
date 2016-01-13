import unirest
import json
from config import model_params, optimization_params, dataset_params, filename_params, visual_params, \
    number_of_epochs, verbose, dataset_path
class ServerCommunication(object):

    def __init__(self):
        self.base_url = "http://127.0.0.1:3000/"
        self.stop = False
        self.default_headers = { "Accept": "application/json", "content-type": "application/json", "data-type": "json" }
        self.current_id = -1


    def get_stop_status(self):
        url = self.base_url + "job/stopstatus/" +  self.current_id
        def callback(response):
            print(response.code)
            self.stop = True
        thread = unirest.get(url, headers=self.default_headers, callback=callback)

    def append_job_update(self, job):
        url = self.base_url + "job/update/" +  self.current_id
        def callback(response):
            print(response.code)
        thread = unirest.post(url, headers=self.default_headers, callback=callback, params=job)

    def start_new_job(self):
        url = self.base_url + "job/start/"
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
            self.current_id = response.body['id']
            
        thread = unirest.post(url, headers=self.default_headers, params=data, callback=callback)

#TODO: finished job endpoint
#TODO: Singleton by not using class.
