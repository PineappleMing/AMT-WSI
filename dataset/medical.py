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


class MyData(Dataset):
    def __init__(self,path):
        super(MyData, self).__init__()
        self.train_slide_path = '/nfs/yuxiaotian/CAMELYON16/training'
        self.medical_all_ctg, self.label = self.build_list()


    def build_list(self):
        normal_ctg = glob.glob(os.path.join(self.train_slide_path, "slide/normal", "normal_*.tif"))
        tumor_ctg =  glob.glob(os.path.join(self.train_slide_path, "slide/tumor", "tumor-*.tif"))
        medical_all_ctg = normal_ctg + tumor_ctg

        tumor_label_ctg = glob.glob(os.path.join(self.train_slide_path, "lesion_annotations", "tumor-*.xml"))
        label = [None] * len(normal_ctg) + tumor_label_ctg
        return medical_all_ctg,label

    def __len__(self):
        normal_ctg = glob.glob(os.path.join(self.train_slide_path, "slide/normal", "normal_*.tif"))
        tumor_ctg =  glob.glob(os.path.join(self.train_slide_path, "slide/tumor", "tumor-*.tif"))

        return len(normal_ctg) + len(tumor_ctg)

    def __getitem__(self, index):

        medical_tag_path = self.medical_all_ctg[index]
        medical_type,medical_type_num = medical_tag_path.split("slide/")[1].split('/')  #'tumor','tumor-001.tif'
        if(medical_type=='tumor'):
            medical_type_num = medical_type_num.split('.')[0] + '.xml'  #tumor-001.xml
            label_path=os.path.join(self.train_slide_path, "lesion_annotations", medical_type_num)
        else:
            label_path=""
        return medical_tag_path,label_path




class MyTestData(Dataset):
    def __init__(self, params):
        super(MyTestData, self).__init__()
        self.test_im_path = params['test_im_path']
        self.test_lb_path = params['test_lb_path']
        self.test_im_num = 10000
        self.test_labels = open(self.test_lb_path, 'r').read().splitlines()

    def __len__(self):
        return self.test_im_num

    def __getitem__(self, index):
        # load image
        img_file = os.path.join(self.test_im_path, str(index) + '.png')
        img = Image.open(img_file)
        im = self.transform(img)

        # load label
        lb = int(self.test_labels[index])

        return im, lb



