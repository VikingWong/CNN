import sys, os
import scipy
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("./"))

import tools.figure.util as util

p_value_threshold = 0.1
sub_folder = '0'
path = '/home/olav/Documents/Results/E1-mass-boot-100'
folders = ['baseline', 'bootstrapping' ]
pr_key_x = 'threshold'
pr_key_y = 'curve'
lc_key_x = 'epoch'
lc_key_y = 'test_loss'

def perform_welchs_test(s1, s2):
    return scipy.stats.ttest_ind(s1, s2, equal_var=False)

print("Calculate welch's t test for two samples.")
print("Null hypothesis: mean of s1 = mean of s2")
print("Alternate hypothesis: Mean of s1 != mean of s2")
all_tests = []
data = {}
nr_tests = 0
lc = {}
pr = {}
for folder in folders:
    paths = os.listdir(os.path.join(path, folder, sub_folder))
    nr_tests += len(paths)
    lc[folder] = []
    pr[folder] = []
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
            lc[folder].append(final_test_loss)
            print(final_test_loss)


for folder in folders:
        print
        print folder
        for d in data[folder]:
            breakeven_points = util.find_breakeven(d['curve'])
            pr[folder].append(breakeven_points[-1])
            print(breakeven_points[-1])

print("Loss samples t test")
#Random samples from normal
s = np.random.normal(np.mean(lc[folders[0]]), np.std(lc[folders[1]]), 100)
print("Random samples", scipy.stats.shapiro(s))
fig = qqplot(s, scipy.stats.norm, fit=True, line='45')
plt.show()

#First folder figures
fig = qqplot(np.array(lc[folders[0]]), scipy.stats.norm, fit=True, line='45')
plt.show()
#scipy.stats.probplot(lc[folders[0]], dist="norm", plot=plt)
plt.show()
print(folders[0], scipy.stats.shapiro(np.array(lc[folders[0]])))

#Second folder figures
fig = qqplot(np.array(lc[folders[1]]), scipy.stats.norm, fit=True, line='45')
plt.show()
#scipy.stats.probplot(lc[folders[1]], dist="norm", plot=plt)
#plt.show()
print(folders[1], scipy.stats.shapiro(np.array(lc[folders[1]])))

tstat, pval = perform_welchs_test(lc[folders[0]], lc[folders[1]])
print("t-statistics = {}".format(tstat))
print("p-value = {}".format(pval) )
if pval < p_value_threshold:
    print("Reject null hypothesis!")
    print("Means unequal")
else:
    print("Do not reject!")
print
print("Breakeven samples t test")
tstat, pval = perform_welchs_test(pr[folders[0]], pr[folders[1]])
print("t-statistics = {}".format(tstat))
print("p-value = {}".format(pval) )
if pval < p_value_threshold:
    print("Reject null hypothesis!")
    print("Means unequal")
else:
    print("Do not reject!")
print