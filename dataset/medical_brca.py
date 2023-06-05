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
        self.label_arr = np.load(self.label_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.img_arr)

    def __getitem__(self, index):
        uuid = self.img_arr[index].split('.')[0]
        return uuid, self.label_arr[uuid]


class Data_BRCA_Test(Dataset):
    def __init__(self):
        super(Data_BRCA_Test, self).__init__()
        self.train_path = "/nfs3-p1/yuxiaotian/BRCA/slide/"
        self.label_path = "/nfs3/yuxiaotian/BRCA/u2l.npy"
        self.img_path = '/nfs3-p1/lhm/BRCA/s256/'
        self.data_arr = os.listdir(self.train_path)
        self.img_arr = os.listdir(self.img_path)[:80]
        self.label_arr = np.load(self.label_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.img_arr)

    def __getitem__(self, index):
        uuid = self.img_arr[index].split('.')[0]
        return uuid, self.label_arr[uuid]


if __name__ == '__main__':
    train_slide_path = '/nfs/yuxiaotian/CAMELYON16/training'
    normal_ctg = glob.glob(os.path.join(train_slide_path, "slide/normal", "normal_*.tif.invalid"))
    tumor_ctg = glob.glob(os.path.join(train_slide_path, "slide/tumor", "tumor-*.tif.tif"))
    for j in normal_ctg:
        j_tag = j.split(".tif.invalid")[0] + '.tif'
        os.system(
            'mv {} {}'.format(j, j_tag))

    # data = MyData('')
    # medical_size = []
    # medical_invalid = []
    # scale = [6,7,8,9]
    #
    # for i in range(235):
    #     a, b = data[i]
    #     for j in scale:
    #         try:
    #             slide = openslide.OpenSlide(a)
    #             height, width = slide.level_dimensions[j]
    #             tick = 'class' + str(j) + 'is valid' + 'size is:' + str(height) +',width:' + str(width)
    #             medical_size.append(tick)
    #             print(a, ':valid')
    #         except:
    #             tick = a + ',class' + str(j)  + 'is invalid'
    #             medical_size.append(tick)
    #             print(a,':invalid')
    #
    # with open('./result_2.txt','w',encoding="utf-8") as fp:
    #     for k in medical_size:
    #         fp.write(k+'\n')
