import glob

import numpy as np

# datasets = ['5_year_no_recur/', 'alive_after_5_years/', 'alive_after_5_years/fine/',
#             'dead_within_2_years/', 'recur_alive_2_years/']
datasets = ['5_year_no_recur/', 'alive_after_5_years/']
# self.datasets = ['dead_within_2_years/', 'recur_alive_2_years/']
label_path = '/home/lhm/Vit/HCC/label/'

label_all = np.load('/nfs3/lhm/HCC/label/label_all.npy')

train_all = label_all[:int(len(label_all) * 0.8)]
test_all = label_all[int(len(label_all) * 0.8):]
pos = 0
neg = 0
for train_ctg in train_all:
    label = np.load(train_ctg, allow_pickle=True)
    if label[0] == 0:
        neg += 1
    else:
        pos += 1
print(pos, neg)
