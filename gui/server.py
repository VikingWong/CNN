import unirest

class ServerCommunication(object):

    def __init__(self):
        self.base_url = "127.0.0.1:1337/"
        self.stop = False
        self.default_headers = { "Accept": "application/json"}
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
        def callback(response):
            self.current_id = response.body.job_id
        thread = unirest.get(url, headers=self.default_headers, callback=callback)

#TODO: finished job endpoint