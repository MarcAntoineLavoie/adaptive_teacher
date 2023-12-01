import torch
import json
import numpy as np
import matplotlib.pyplot as plt

filename = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/at_scaled/metrics.json'
with open(filename, 'r') as f_in:
    data = json.load(f_in)

datasets = ['cityscapes_val', 'cityscapes_foggy_val', 'cityscapes_foggy_train_weak', 'cityscapes_foggy_train_strong', 'cityscapes_foggy_pseudo_strong']
classes = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']
AP = ['AP','AP50','AP75','APs','APm','APl']

ap_splits = np.zeros((len(datasets),len(AP)))
ap_per_class = np.zeros((len(datasets),len(classes)))

ap_splits = np.zeros((len(datasets),len(AP)))
ap_per_class = np.zeros((len(datasets),len(classes)))

for id1, dataset in enumerate(datasets):
    for id2, ap in enumerate(AP):
        key_name = dataset + '/bbox/' + ap
        ap_splits[id1, id2] = data[key_name]
    for id3, class_ in enumerate(classes):
        key_name = dataset + '/bbox/AP50-' + class_
        ap_per_class[id1, id3] = data[key_name]

N = len(AP)
ind = np.arange(N) 
width = 1/(len(datasets)+2)

plt.figure()
for i in range(len(datasets)):
    vals = ap_splits[i,:]
    plt.bar(ind+width*i, vals, width)
  
plt.xlabel("Measure")
plt.ylabel('Score')
plt.xticks(ind+2*width, AP)
plt.legend(datasets)
plt.tight_layout()


N = len(classes)
ind = np.arange(N) 
width = 1/(len(datasets)+2)
plt.figure()
for i in range(len(datasets)):
    vals = ap_per_class[i,:]
    plt.bar(ind+width*i, vals, width)
  
plt.xlabel("Class")
plt.ylabel('Score')
plt.xticks(ind+2*width, classes)
plt.legend(datasets)
plt.tight_layout()


plt.figure()
fig, ax = plt.subplots()
ax.set_xticks(ind + width / 2, labels=classes)
ax.set_yscale('log')

vals = [3675, 379, 26963, 737, 17919, 1781, 168, 484]
b = ax.bar(ind, vals, 1/5)
ax.set_xlabel("Class")
ax.set_ylabel('Instance Count')
plt.xticks(ind, classes)
plt.tight_layout()

plt.show()

plt.figure();plt.plot(ap_per_class.T);plt.legend(datasets)
plt.figure();plt.plot(ap_splits.T);plt.legend(datasets)
plt.figure();plt.plot((ap_per_class/ap_per_class[0,:]).T);plt.legend(datasets)
plt.figure();plt.plot((ap_splits/ap_splits[0,:]).T);plt.legend(datasets)
plt.show()


import matplotlib.pyplot as plt
import json
import numpy as np

# file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/test_iou_select_080/results_big.json'
file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/test_v2_iou70/results.json'
tf_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/test_v2_iou70/tf_out.json'
with open(file_in, 'r') as f_in:
    results = json.load(f_in)

# with open(tf_in, 'r') as f_in:
#     results_tf1 = json.load(f_in)

dict_class = dict([(x.split('-')[1], []) for x in results['cityscapes_val']['bbox'].keys() if '-' in x])
run_order = []
# labels = list(dict_class.keys())
for run in results.keys():
    run_order.append(run)
    for label in results[run]['bbox'].keys():
        if '-' in label:
            curr_class = label.split('-')[1]
            dict_class[curr_class].append(results[run]['bbox'][label])

vals = np.array(list(dict_class.values()))
order = [3,4,0,1,2]
order2 = np.argsort(list(dict_class.keys()))
vals = vals[:,order]
vals = vals[order2,:]
class_labels = [list(dict_class.keys())[x] for x in [7, 4, 2, 6, 0, 1, 5, 3]]
run_labels = [run_order[x] for x in order]

N = len(dict_class)
ind = np.arange(N) 
width = 1/(len(order)+2)
plt.figure()
for i in range(len(order)):
    plt.bar(ind+width*i, vals[:,i], width)
  
plt.xlabel("Class")
plt.ylabel('Score')
plt.xticks(ind+2*width, class_labels)
plt.legend(run_labels)
plt.title("AP50 New")
plt.tight_layout()

# file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/at_scaled/metrics.json'
# file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/at_scaled/results.json'
file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/test_v2_nom/results.json'
tf_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/test_v2_nom/tf_out.json'
with open(file_in, 'r') as f_in:
    temp = json.load(f_in)

# with open(tf_in, 'r') as f_in:
#     results_tf2 = json.load(f_in)

# old_vals = np.zeros((8,5))
# order_old = [4,3,2,1,0]
# i = 0
# j = -1
# old_key = None
# for key in temp.keys():
#     if key[0] == 'c' and '-' in key:
#         curr_key = key.split('/')[0]
#         if old_key != curr_key:
#             old_key = curr_key
#             j += 1
#             i = 0
#         old_vals[i,j] = temp[key]
#         i += 1

# old_vals = old_vals[:,order_old]

# file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/at_scaled/results.json'
# with open(file_in, 'r') as f_in:
#     temp = json.load(f_in)

dict_class = dict([(x.split('-')[1], []) for x in temp['cityscapes_val']['bbox'].keys() if '-' in x])
run_order = []
# labels = list(dict_class.keys())
for run in temp.keys():
    run_order.append(run)
    for label in temp[run]['bbox'].keys():
        if '-' in label:
            curr_class = label.split('-')[1]
            dict_class[curr_class].append(temp[run]['bbox'][label])

old_vals = np.array(list(dict_class.values()))
order = [3,4,0,1,2]
order2 = np.argsort(list(dict_class.keys()))
old_vals = old_vals[:,order]
old_vals = old_vals[order2,:]
class_labels = [list(dict_class.keys())[x] for x in [7, 4, 2, 6, 0, 1, 5, 3]]
run_labels = [run_order[x] for x in order]

N = len(dict_class)
ind = np.arange(N) 
width = 1/(len(order)+2)
plt.figure()
for i in range(len(order)):
    plt.bar(ind+width*i, old_vals[:,i], width)
  
plt.xlabel("Class")
plt.ylabel('Score')
plt.xticks(ind+2*width, class_labels)
plt.legend(run_labels)
plt.title("AP50 Baseline")
plt.tight_layout()

N = len(dict_class)
ind = np.arange(N) 
width = 1/(len(order)+2)
plt.figure()
for i in range(len(order)):
    plt.bar(ind+width*i, vals[:,i] - old_vals[:,i], width)
  
plt.xlabel("Class")
plt.ylabel('Score')
plt.xticks(ind+2*width, class_labels)
plt.legend(run_labels)
plt.title("AP50 New - Baseline")
plt.tight_layout()

a = 1