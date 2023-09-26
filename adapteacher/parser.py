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


