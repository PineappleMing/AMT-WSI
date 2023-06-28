import random

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


class Data_HCC(Dataset):
    def __init__(self):
        super(Data_HCC, self).__init__()
        self.root_path = "/medical-data/yxt/WSI/Hepatoma_first_trial/"
        self.datasets = ['5_year_no_recur/', 'alive_after_5_years/', 'dead_within_2_years/', 'recur_alive_2_years/']
        # self.datasets = ['dead_within_2_years/', 'recur_alive_2_years/']
        self.label_path = '/home/lhm/Vit/HCC/label/'
        self.label_all = self.build_list()

    def build_list(self):
        label_all = []
        for dataset in self.datasets:
            paths = glob.glob(self.label_path + dataset + '*.npy')
            label_all += paths
        random.shuffle(label_all)

        label_all = label_all[:int(len(label_all) * 0.8)]
        return label_all

    def __len__(self):
        return len(self.label_all)

    def __getitem__(self, index):
        filename = self.label_all[index].split('/')[-1]
        dataset = self.label_all[index].split('/')[-2]
        code = filename.split('.')[0]
        medical_tag = self.root_path + dataset + '/' + code + '.svs'
        return medical_tag, self.label_all[index]


class Data_HCC_Test(Dataset):
    def __init__(self):
        super(Data_HCC_Test, self).__init__()
        self.root_path = "/medical-data/yxt/WSI/Hepatoma_first_trial/"
        self.datasets = ['5_year_no_recur/', 'alive_after_5_years/', 'dead_within_2_years/', 'recur_alive_2_years/']
        self.label_path = '/home/lhm/Vit/HCC/label/'
        self.label_all = self.build_list()

    def build_list(self):
        label_all = []
        for dataset in self.datasets:
            paths = glob.glob(self.label_path + dataset + '*.npy')
            label_all += paths
        random.shuffle(label_all)
        label_all = label_all[420:]
        return label_all

    def __len__(self):
        return len(self.label_all)

    def __getitem__(self, index):
        filename = self.label_all[index].split('/')[-1]
        dataset = self.label_all[index].split('/')[-2]
        code = filename.split('.')[0]
        medical_tag = self.root_path + dataset + '/' + code + '.svs'
        return medical_tag, self.label_all[index]
