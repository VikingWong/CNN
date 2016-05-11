import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import json
font = {'size'   : 15}

matplotlib.rc('font', **font)

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
    fig.tight_layout()
    fig.savefig('pr.png')
    plt.show()


def display_loss_curve_plot(series):
    fig, ax = plt.subplots()
    #plt.suptitle('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    for serie in series:
        ax.plot([p['epoch'] for p in serie['data'][1:]], [p[serie["y_key"]] for p in serie['data'][1:]],
                label=serie['name'].capitalize())
    ax.legend(loc='upper right', shadow=True)
    fig.tight_layout()
    fig.savefig('pr.png')
    plt.show()

def display_two_axis_plot(series, axis2_serie):
    fig, ax = plt.subplots()
    color_cycle = ax._get_lines.color_cycle
    plt.grid(True)
    ax2 = ax.twinx()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax2.set_ylabel('Precision and recall breakeven')
    for serie in series:
        ax.plot([p['epoch'] for p in serie['data'][1:]], [p[serie["y_key"]] for p in serie['data'][1:]],
                label=serie['name'].capitalize(),
                color=next(color_cycle))

    for serie in axis2_serie:
        color = next(color_cycle) #Color used for both legend fix and plot color
        ax2.plot([p['epoch'] for p in serie['data'][1:]], [p[serie["y_key"]] for p in serie['data'][1:]],
                 marker='o',
                 label=serie['name'].capitalize(),
                 color=color)
        ax.plot(0, 0, label = serie['name'].capitalize(), color=color, marker='o') #Adds legend to ax1, fix
    ax.legend(loc='lower right', shadow=True)
    fig.tight_layout()
    plt.show()

def display_noise_summary(series, x_label, y_label):
    fig, ax = plt.subplots()
    #plt.suptitle('Loss curve')
    plt.xlabel(x_label.capitalize())
    plt.ylabel(y_label.capitalize())
    plt.grid(True)
    marker = ['s', 'v', 'o', '^', '<', '>']
    for i, serie in enumerate(series):
        ax.plot([p["x"] for p in serie['data']], [p["y"] for p in serie['data']], label=serie['name'].capitalize(),
                marker=marker[i%len(marker)], ms=8.0)
    ax.legend(loc='upper right', shadow=True)
    fig.tight_layout()
    fig.savefig('summary.png')
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


def find_breakeven(pr, samples=14):
    #Recall levels that are 0 get a penalty in the sorter
    temp = sorted(pr, key=lambda p: abs(p['precision'] - p['recall'] + int(not bool(p['recall']))))
    temp2 = temp[0: samples :]
    points = sorted(temp2, key=lambda p: p['recall'])
    x = np.array([v['recall'] for v in points])
    y = np.array([v['precision'] for v in points])
    poly_coeff = np.polynomial.polynomial.polyfit(x, y, 2)
    roots = np.polynomial.polynomial.polyroots(poly_coeff - [0, 1, 0])
    return roots


def open_json_result(file_path):
    data = {}
    with open(file_path) as data_file:
        data = json.load(data_file)
    return data
