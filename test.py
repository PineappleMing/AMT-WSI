import glob
import os

import cv2
import numpy as np
import openslide
import tqdm
from PIL import Image

datasets = ['dead_within_2_years/', 'recur_alive_2_years/']

root_path = '/home/lhm/Vit/HCC/label/'
pos = 0
neg = 0
for d in datasets:
    us = glob.glob(root_path + d + '*.npy')
    for u in us:
        a = np.load(u, allow_pickle=True)
        if a[0]:
            pos += 1
        else:
            neg += 1

print(pos)
print(neg)
