import sys, os

sys.path.append(os.path.abspath("./"))

import tools.figure.util as util


sub_folder = ''
path = '/home/olav/Documents/Results/E7_inexperienced_teacher'
folders = ['baseline - gradual', 'curriculum - gradual' ]
pr_key_x = 'threshold'
pr_key_y = 'curve'
lc_key_x = 'epoch'
lc_key_y = 'test_loss'


print("Creating comparison figures")
all_tests = []
data = {}
nr_tests = 0
for folder in folders:
    paths = os.listdir(os.path.join(path, folder, sub_folder))
    nr_tests += len(paths)
    print("Folder", folder, "length", len(paths))
    all_tests.append(paths)
    data[folder] = []

for t in range(len(all_tests)):
    for data_path in all_tests[t]:
        json_data = util.open_json_result(os.path.join(path, folders[t], sub_folder, data_path))

        if type(json_data) is list:
            d = json_data[0]
        else:
            d = json_data
        data[folders[t]].append(d)

for folder in folders:
        print
        print folder
        for d in data[folder]:
                final_test_loss = d['events'][-1][lc_key_y]
                print(final_test_loss)

for folder in folders:
        print
        print folder
        for d in data[folder]:
                #Verify visually
                breakeven_points = util.find_breakeven(d['curve'])
                print(breakeven_points[-1])