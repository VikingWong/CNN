import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import json

def display_precision_recall_plot(series):

    fig, ax = plt.subplots()
    #plt.suptitle('Precision and recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    for serie in series:
        ax.plot([p['recall'] for p in serie['data']], [p['precision'] for p in serie['data']], label=serie['name'].capitalize())
        if serie['breakeven'] is not None:
            print(serie['breakeven'])
            ax.plot(serie['breakeven'][-1], serie['breakeven'][-1] , 'bo', ms=3.5, mfc="black")
    ax.legend(loc='lower left', shadow=True)
    plt.show()


def display_loss_curve_plot(series):
    fig, ax = plt.subplots()
    #plt.suptitle('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    for serie in series:
        ax.plot([p['epoch'] for p in serie['data'][1:]], [p[serie["y_key"]] for p in serie['data'][1:]], label=serie['name'].capitalize())
    ax.legend(loc='upper right', shadow=True)
    plt.show()

def display_noise_summary(series, x_label, y_label):
    fig, ax = plt.subplots()
    #plt.suptitle('Loss curve')
    plt.xlabel(x_label.capitalize())
    plt.ylabel(y_label.capitalize())
    plt.grid(True)
    marker = ['s', 'v', 'o', '^', '<', '>']
    for i, serie in enumerate(series):
        ax.plot([p["x"] for p in serie['data']], [p["y"] for p in serie['data']], label=serie['name'].capitalize(), marker=marker[i%len(marker)], ms=8.0)
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
                if j < len(series[s][series_key]):
                    values.append(series[s][series_key][j][k])
            #print(k)
            #print(sum(values)/len(values))
            combined[j][k] = sum(values)/len(values)
    return combined


def find_breakeven(pr):
    temp = sorted(pr, key=lambda p: abs(p['precision'] - p['recall']))
    temp2 = temp[0: 14 :]
    points = sorted(temp2, key=lambda p: p['recall'])
    x = np.array([v['recall'] for v in points])
    y = np.array([v['precision'] for v in points])
    #print(x)
    #print(y2)
    poly_coeff = np.polynomial.polynomial.polyfit(x, y, 2)
    roots = np.polynomial.polynomial.polyroots(poly_coeff - [0, 1, 0])
    return roots


def open_json_result(file_path):
    data = {}
    with open(file_path) as data_file:
        data = json.load(data_file)
    return data
