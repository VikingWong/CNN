import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import json

def display_precision_recall_plot(series):

    fig, ax = plt.subplots()
    plt.suptitle('Precision and recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    for serie in series:
        ax.plot([p['recall'] for p in serie['data']], [p['precision'] for p in serie['data']], label=serie['name'])
    ax.legend(loc='lower left', shadow=True)
    plt.show()


def display_loss_curve_plot(series):
    fig, ax = plt.subplots()
    #plt.suptitle('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    for serie in series:
        ax.plot([p['epoch'] for p in serie['data'][1:]], [p[serie["y_key"]] for p in serie['data'][1:]], label=serie['name'])
    ax.legend(loc='upper right', shadow=True)
    plt.show()


def average(series, series_key, x_align_key):
    #Assume that all series, and datapoints contain the same keys. Everyting is recorded.
    if len(series) <= 0:
        return []
    nr_datapoints = len(series[0][series_key])
    if nr_datapoints <= 0:
        return []

    keys = series[0][series_key][0].keys()

    #TODO: better way to avoid not summable keys? Check type of each value
    if'date_recorded' in keys:
        d = keys.index('date_recorded')
        del keys[d]

    if'training_rate' in keys:
        d = keys.index('training_rate')
        del keys[d]

    combined = []
    for i in range(nr_datapoints):
        combined.append({})

    for k in keys:
        for j in range(nr_datapoints):
            values = []
            for s in range(len(series)):
                values.append(series[s][series_key][j][k])
            #print(k)
            #print(sum(values)/len(values))
            combined[j][k] = sum(values)/len(values)
    return combined


def open_json_result(file_path):
    data = {}
    with open(file_path) as data_file:
        data = json.load(data_file)
    return data
