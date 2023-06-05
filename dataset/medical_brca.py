import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import glob
# from utils import *
import openslide


class Data_BRCA(Dataset):
    def __init__(self):
        super(Data_BRCA, self).__init__()
        self.train_path = "/nfs3-p1/yuxiaotian/BRCA/slide/"
        self.label_path = "/nfs3/yuxiaotian/BRCA/u2l.npy"
        self.img_path = '/nfs3-p1/lhm/BRCA/s256/'
        self.data_arr = os.listdir(self.train_path)
        self.img_arr = os.listdir(self.img_path)
        self.uid_arr = []
        self.label_arr = np.load(self.label_path, allow_pickle=True).item()
        for i in self.img_arr:
            uid = i.split('.')[0]
            if uid in list(self.label_arr.keys()):
                self.uid_arr.append(uid)
        print(len(self.img_arr), len(self.uid_arr))

    def __len__(self):
        return len(self.uid_arr)

    def __getitem__(self, index):
        uuid = self.uid_arr[index]
        return uuid, self.label_arr[uuid]


class Data_BRCA_Test(Dataset):
    def __init__(self):
        super(Data_BRCA_Test, self).__init__()
        self.train_path = "/nfs3-p1/yuxiaotian/BRCA/slide/"
        self.label_path = "/nfs3/yuxiaotian/BRCA/u2l.npy"
        self.img_path = '/nfs3-p1/lhm/BRCA/s256/'
        self.data_arr = os.listdir(self.train_path)
        self.img_arr = os.listdir(self.img_path)[600:]
        self.uid_arr = []
        self.label_arr = np.load(self.label_path, allow_pickle=True).item()
        for i in self.img_arr:
            uid = i.split('.')[0]
            if uid in list(self.label_arr.keys()):
                self.uid_arr.append(uid)

    def __len__(self):
        return len(self.uid_arr)

    def __getitem__(self, index):
        uuid = self.uid_arr[index]
        return uuid, self.label_arr[uuid]
