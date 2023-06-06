import glob

import cv2
import numpy as np
import openslide
import torch
from torch import Tensor
from torchvision.transforms import transforms

from dataloader_adapters.InputTensorAdapter import InputTensorAdapter

mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
t_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


class LUADInputTensorAdapter(InputTensorAdapter):
    __slots__ = ['labels']

    def __init__(self, data_paths: list, label_paths: list, rate: int = 8, num_focus: int = 4,
                 num_patch_sqrt: int = 16) -> None:
        super().__init__(data_paths, label_paths, rate, num_focus, num_patch_sqrt)
        self.labels = self.label_paths

    def getStageOneInputTensor(self) -> (Tensor, Tensor, Tensor):
        slide_all = []
        for uuid in self.data_paths:
            try:
                img_name = "/nfs3-p1/lhm/LUAD/s64/" + uuid + '.png'
                img = cv2.imread(img_name)
                slide_all.append(t_eval(img).unsqueeze(0))
            except Exception as E:
                print(uuid, "is invalid", E)
        inputs1 = torch.cat(slide_all, dim=0).float().to(self.device)  # [16,3,256,256]
        stage_one_label_all = self.labels.to(self.device)
        stage_one_label = self.labels.unsqueeze(1).repeat(1, 64).to(self.device)
        return inputs1, stage_one_label_all, stage_one_label

    def getStageTwoInputTensor(self, focus_index) -> (Tensor, Tensor, Tensor):
        slide_all = []
        B, P1, P2 = len(self.label_paths), self.num_focus, self.num_patch_sqrt * self.num_patch_sqrt
        stage_two_label = torch.zeros((B, P1, P2), dtype=torch.long).to(self.device)
        stage_two_label_all = torch.zeros((B, P1), dtype=torch.long).to(self.device)

        for i, uuid in enumerate(self.data_paths):
            try:
                img16 = cv2.imread("/nfs3-p1/lhm/LUAD/s8/" + uuid + '.png')
                x_1 = focus_index[i] % self.num_patch_sqrt
                y_1 = focus_index[i] // self.num_patch_sqrt
                size_1 = 256
                for j in range(self.num_focus):
                    cord_x = size_1 * x_1[j]
                    cord_y = size_1 * y_1[j]
                    patch = img16[cord_y:cord_y + size_1, cord_x:cord_x + size_1, :]
                    slide_all.append(t_eval(patch).unsqueeze(0))
                    stage_two_label_all[i, j] = self.labels[i]
                    for k in range(self.num_patch_sqrt * self.num_patch_sqrt):
                        stage_two_label[i, j, k] = self.labels[i]
            except Exception as E:
                print(i, "is invalid", E)
        inputs2 = torch.cat(slide_all, dim=0).float().to(self.device)
        return inputs2, stage_two_label_all, stage_two_label

    def getStageThreeInputTensor(self, focus_index1, focus_index2) -> (Tensor, Tensor):
        slide_all = []
        B = len(self.label_paths)
        stage_three_label_all = torch.zeros((B, self.num_focus, self.num_focus),
                                            dtype=torch.long).to(self.device)
        for i, uuid in enumerate(self.data_paths):
            try:
                slide_name = glob.glob('/nfs3-p1/yuxiaotian/LUAD/slide/' + uuid + '/*.svs*')[0]
                slide = openslide.OpenSlide(slide_name)
                W, H = slide.level_dimensions[0]
                ori_x = W // 2 - 256 * self.rate * self.rate // 2
                ori_y = H // 2 - 256 * self.rate * self.rate // 2
                x_1 = focus_index1[i] % self.num_patch_sqrt
                y_1 = focus_index1[i] // self.num_patch_sqrt
                x_2 = focus_index2 % self.num_patch_sqrt
                y_2 = focus_index2 // self.num_patch_sqrt
                size_1 = 256 * self.num_patch_sqrt
                size_2 = 256
                for j in range(self.num_focus):
                    for k in range(self.num_focus):
                        cord_x = ori_x + size_1 * x_1[j] + size_2 * x_2[self.num_focus * i + j, k]
                        cord_y = ori_y + size_1 * y_1[j] + size_2 * y_2[self.num_focus * i + j, k]
                        img = slide.read_region((cord_x, cord_y), 0,
                                                (256,
                                                 256)).convert('RGB')
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        slide_all.append(t_eval(img).unsqueeze(0))
                        stage_three_label_all[i, j, k] = self.labels[i]
            except Exception as E:
                print(i, "is invalid", E)
        inputs3 = torch.cat(slide_all, dim=0).float().to(self.device)

        return inputs3, stage_three_label_all
