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


class Data_Cam(Dataset):
    def __init__(self):
        super(Data_Cam, self).__init__()
        self.train_slide_path = '/nfs3-p1/yuxiaotian/CAMELYON16/training'
        self.medical_all_ctg, self.label = self.build_list()

    def build_list(self):
        normal_ctg = glob.glob(os.path.join(self.train_slide_path, "slide/normal", "normal_*.tif"))
        tumor_ctg = glob.glob(os.path.join(self.train_slide_path, "slide/tumor", "tumor-*.tif"))
        medical_all_ctg = tumor_ctg + normal_ctg

        tumor_label_ctg = glob.glob(os.path.join("/nfs3-p1/lhm/CAMEL/label/tumor", "tumor-*.npy"))
        label = tumor_label_ctg + [None] * len(normal_ctg)
        return medical_all_ctg, label

    def __len__(self):
        normal_ctg = glob.glob(os.path.join(self.train_slide_path, "slide/normal", "normal_*.tif"))
        tumor_ctg = glob.glob(os.path.join(self.train_slide_path, "slide/tumor", "tumor-*.tif"))

        return len(normal_ctg) + len(tumor_ctg)

    def __getitem__(self, index):
        medical_tag_path = self.medical_all_ctg[index]
        medical_type, medical_type_num = medical_tag_path.split("slide/")[1].split('/')  # 'tumor','tumor-001.tif'
        if (medical_type == 'tumor'):
            medical_type_num = medical_type_num.split('.')[0] + '.npy'  # tumor-001.xml
            label_path = os.path.join("/nfs3-p1/lhm/CAMEL/label/tumor", medical_type_num)
        else:
            label_path = ""
        return medical_tag_path, label_path


class Data_Cam_Test(Dataset):
    def __init__(self):
        super(Data_Cam_Test, self).__init__()
        self.test_slide_path = '/nfs/yuxiaotian/CAMELYON16/testing'
        self.medical_all_ctg = self.build_list()

    def build_list(self):
        medical_ctg = glob.glob(os.path.join(self.test_slide_path, "images", "test_*.tif"))

        return medical_ctg

    def __len__(self):
        return len(self.medical_all_ctg)

    def __getitem__(self, index):
        medical_tag_path = self.medical_all_ctg[index]
        filename = medical_tag_path.split('/')[-1].split('.')[0]
        label_path = "/nfs3-p1/lhm/CAMEL/label/test/" + filename + '.npy'
        if not os.path.exists(label_path):
            label_path = ''
        return medical_tag_path, label_path


if __name__ == '__main__':
    train_slide_path = '/nfs/yuxiaotian/CAMELYON16/training'
    normal_ctg = glob.glob(os.path.join(train_slide_path, "slide/normal", "normal_*.tif.invalid"))
    tumor_ctg = glob.glob(os.path.join(train_slide_path, "slide/tumor", "tumor-*.tif.tif"))
    for j in normal_ctg:
        j_tag = j.split(".tif.invalid")[0] + '.tif'
        os.system(
            'mv {} {}'.format(j, j_tag))

