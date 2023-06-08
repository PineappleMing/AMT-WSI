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
train_panda_6_path = '/nfs/guojun/panda-6-n/'
train_panda_3_path = '/nfs/guojun/panda-3/'
tiff_panda_path = '/nfs3-p1/yuxiaotian/PANDA/train_slide/'
tiff_panda_mask_path = '/nfs3-p1/yuxiaotian/PANDA/train_label_masks/'


class PandaInputTensorAdapter(InputTensorAdapter):
    __slots__ = ['labels']

    def __init__(self, data_paths: list, label_paths: list, rate: int = 8, num_focus: int = 4,
                 num_patch_sqrt: int = 16) -> None:
        super().__init__(data_paths, label_paths, rate, num_focus, num_patch_sqrt)
        self.labels = torch.zeros(len(label_paths))

    def getStageOneInputTensor(self) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        slide_all = []
        B, P1 = len(self.label_paths), self.num_patch_sqrt * self.num_patch_sqrt
        stage_one_true_label = torch.zeros((B, P1), dtype=torch.long).to(self.device).detach()
        for i, path in enumerate(self.data_paths):
            try:
                slide_name = path.split('/')[-1].split('.')[0]
                img_name = train_panda_6_path + slide_name + '.jpg'
                mask_name = train_panda_6_path + slide_name + '_mask.png'
                img = cv2.imread(img_name)
                mask = cv2.imread(mask_name)
                slide_all.append(t_eval(img).unsqueeze(0))
                label = mask.max()
                self.labels[i] = int(label > 2)
                for j in range(P1):
                    cord_y, cord_x = j * 32 // self.num_patch_sqrt, j * 32 % self.num_patch_sqrt
                    patch_mask = mask[cord_y:cord_y + 32, cord_x:cord_x + 32]
                    stage_one_true_label[i, j] = int(patch_mask.max() > 2)
            except Exception as E:
                print(path, "is invalid", E)
        inputs1 = torch.cat(slide_all, dim=0).float().to(self.device)  # [16,3,256,256]
        stage_one_label_all = self.labels.long().to(self.device)
        stage_one_label = self.labels.unsqueeze(1).repeat(1, 64).long().to(self.device)
        stage_one_true_label_all = stage_one_label_all.clone().detach()
        return inputs1, stage_one_label_all, stage_one_label, stage_one_true_label_all, stage_one_true_label

    def getStageTwoInputTensor(self, focus_index) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        slide_all = []
        B, P1, P2 = len(self.label_paths), self.num_focus, self.num_patch_sqrt * self.num_patch_sqrt
        stage_two_label = torch.zeros((B, P1, P2), dtype=torch.long).to(self.device)
        stage_two_label_all = torch.zeros((B, P1), dtype=torch.long).to(self.device)
        stage_two_true_label = torch.zeros((B, P1, P2), dtype=torch.long).to(self.device).detach()
        stage_two_true_label_all = torch.zeros((B, P1), dtype=torch.long).to(self.device).detach()
        for i, path in enumerate(self.data_paths):
            try:
                slide_name = path.split('/')[-1].split('.')[0]
                img_name = train_panda_3_path + slide_name + '.jpg'
                mask_name = train_panda_6_path + slide_name + '_mask.png'
                img = cv2.imread(img_name)
                mask = cv2.imread(mask_name)
                x_1 = focus_index[i] % self.num_patch_sqrt
                y_1 = focus_index[i] // self.num_patch_sqrt
                size_1 = 256
                for j in range(self.num_focus):
                    cord_x = size_1 * x_1[j]
                    cord_y = size_1 * y_1[j]
                    patch = img[cord_y:cord_y + size_1, cord_x:cord_x + size_1, :]
                    slide_all.append(t_eval(patch).unsqueeze(0))
                    stage_two_label_all[i, j] = self.labels[i]
                    for k in range(self.num_patch_sqrt * self.num_patch_sqrt):
                        stage_two_label[i, j, k] = self.labels[i]
                        y_2, x_2 = k // self.num_patch_sqrt, k % self.num_patch_sqrt
                        h = y_1[j] * 32 + y_2 * 4
                        w = x_1[j] * 32 + x_2 * 4
                        patch_mask = mask[h:h + 4, w:w + 4]
                        stage_two_true_label[i, j, k] = patch_mask.max()
                    stage_two_true_label_all[i, j] = stage_two_true_label[i, j].max()
            except Exception as E:
                print(i, "is invalid", E)

        stage_two_true_label_all = (stage_two_true_label_all > 2).long()
        stage_two_true_label = (stage_two_true_label > 2).long()
        inputs2 = torch.cat(slide_all, dim=0).float().to(self.device)
        return inputs2, stage_two_label_all, stage_two_label, stage_two_true_label_all, stage_two_true_label

    def getStageThreeInputTensor(self, focus_index1, focus_index2) -> (Tensor, Tensor, Tensor):
        slide_all = []
        B = len(self.label_paths)
        stage_three_label_all = torch.zeros((B, self.num_focus, self.num_focus),
                                            dtype=torch.long).to(self.device)
        stage_three_true_label_all = torch.zeros((B, self.num_focus, self.num_focus),
                                                 dtype=torch.long).to(self.device).detach()
        for i, path in enumerate(self.data_paths):
            try:
                slide_name = path.split('/')[-1].split('.')[0]
                img_name = tiff_panda_path + slide_name + '.tiff'
                mask_name = tiff_panda_mask_path + slide_name + '_mask.tiff'
                slide = openslide.OpenSlide(img_name)
                full_mask = openslide.OpenSlide(mask_name)
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
                        mask = full_mask.read_region((cord_x, cord_y), 0,
                                                     (256, 256)).convert('RGB')
                        mask = torch.from_numpy(np.array(mask))
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        slide_all.append(t_eval(img).unsqueeze(0))
                        stage_three_label_all[i, j, k] = self.labels[i]
                        stage_three_true_label_all[i, j, k] = (mask.max() > 2).long()
            except Exception as E:
                print(i, "is invalid", E)
        inputs3 = torch.cat(slide_all, dim=0).float().to(self.device)
        return inputs3, stage_three_label_all, stage_three_true_label_all
