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


train_file = 'train_list.txt'
valid_file = 'valid_list.txt'


class Data_Panda(Dataset):
    def __init__(self):
        super(Data_Panda, self).__init__()
        self.train_path = '/nfs3-p2/yuxiaotian/PANDA'
        self.medical_ctg = self.build_list()


    def build_list(self):
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                medical_ctg = f.read().split()
        else:
            medical_ctg_all = glob.glob(os.path.join(self.train_path, "train_slide", "*.tiff"))
            medical_ctg_train = np.random.choice(medical_ctg_all, len(medical_ctg_all)-200, replace=False)
            medical_ctg_valid = list(set(medical_ctg_all).difference(set(medical_ctg_train)))
            with open(train_file, 'w') as f:
                f.write('\n'.join(medical_ctg_train))
            with open(valid_file, 'w') as f:
                f.write('\n'.join(medical_ctg_valid))
            medical_ctg = medical_ctg_train

        return medical_ctg

    def __len__(self):

        return len(self.medical_ctg)

    def __getitem__(self, index):

        medical_tag_path = self.medical_ctg[index]
        medical_tag_label = medical_tag_path.split('/')[-1].split(".tiff")[0]   #'cd248eb09a3e4a1ce70b2745f45ce332
        label_path = self.train_path +'/train_label_masks/' + medical_tag_label + "_mask.tiff"

        return medical_tag_path,label_path


class Data_Panda_Test(Dataset):
    def __init__(self):
        super(Data_Panda_Test, self).__init__()
        self.train_path = '/nfs3-p2/yuxiaotian/PANDA'
        self.medical_ctg = self.build_list()


    def build_list(self):
        with open(valid_file, 'r') as f:
            medical_ctg = f.read().split()

        return medical_ctg

    def __len__(self):

        return len(self.medical_ctg)

    def __getitem__(self, index):

        medical_tag_path = self.medical_ctg[index]
        medical_tag_label = medical_tag_path.split('/')[-1].split(".tiff")[0]
        label_path = self.train_path +'/train_label_masks/' + medical_tag_label + "_mask.tiff"
        return medical_tag_path,label_path



