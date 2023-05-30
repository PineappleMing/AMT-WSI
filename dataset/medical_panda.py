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
        medical_tag_label = medical_tag_path.split('/')[-1].split(".tiff")[0]   #'cd248eb09a3e4a1ce70b2745f45ce332
        label_path = self.train_path +'/train_label_masks/' + medical_tag_label + "_mask.tiff"

        return medical_tag_path,label_path







if __name__ == '__main__':
    mydata = Data_Panda()
    for i in [45,67,23,53, 99,101, 2]:
        tag, label = mydata[i]
        print(tag, label)

    import openslide
    import numpy
    import cv2
    # import matploblib.plt as plt
    import matplotlib.pyplot as plt

    slide = openslide.OpenSlide(
        r"/nfs3-p2/yuxiaotian/PANDA/train_label_masks/cca735c397880e88192e97d68b97754e_mask.tiff")
    level_count = slide.level_count
    # print ('level_count = ',level_count)
    # [m,n] = slide.dimensions #得出高倍下的（宽，高）
    # print(m,n)
    # #级别k，且k必须是整数，下采样因子和k有关
    for i in range(level_count):
        [m, n] = slide.level_dimensions[i]  # 每一个级别对应的长和宽
        slide_level_downsamples = slide.level_downsamples[i]  # 下采样因子对应一个倍率
        print("k={0}时的长和宽{1}, {2}和下采样倍率{3}".format(i, m, n, slide_level_downsamples))
    slide_downsamples = slide.get_best_level_for_downsample(2.0)  # 选定倍率返回下采样级别
    # print (slide_downsamples)
    [m, n] = slide.level_dimensions[1]
    import time

    start = time.time()

    mask_shape = numpy.zeros((int(m), int(n)), dtype=int)
    # mask_shape_2 = numpy.zeros(m, n)
    tile = numpy.array(slide.read_region((0, 0), 1, (m, n)))[:, :, :3].reshape(-1)
    tile = set(tile)
    tile_2 = tile.sum(2)

    # dets = numpy.array([[1,2],[3,4]])
    # numpy.savetxt("./test.csv", tile,fmt='%f',delimiter=',')
    I, J, K = tile.shape
    # for i in range(I):
    #     for j in range(J):
    #         for k in range(K):
    #             if(tile[i,j,k] != 0):
    #                     mask_shape[j,i] = 1

    numpy.savetxt("./test_2.csv", tile_2, fmt='%f', delimiter=',')
    # print(tile)
    print('耗时', time.time() - start)
    # plt.figure()
    # plt.imshow(tile)
    # cv2.imwrite('./test_2.jpg', tile)
    # plt.show()
    #
    print("test")
    # 上述代码可以得到左上角坐标（0,0）,6级别下，大小是（1528，3432）的图