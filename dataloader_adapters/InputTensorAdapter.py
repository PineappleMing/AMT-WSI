import abc
from abc import ABC, abstractmethod

import torch.cuda
from torch import Tensor


class InputTensorAdapter(ABC):
    '''
    此类为既存torch_dataloader的装饰器，用于进一步处理由dataloader产生的数据和标签，返回用于训练的tensor
    '''
    __slots__ = ['data_paths', 'label_paths', 'thumbnail_path_1', 'thumbnail_path_2', 'device', 'rate', 'num_focus',
                 'num_patch_sqrt']

    def __init__(self, data_paths: list, label_paths: list, rate: int = 8,
                 num_focus: int = 4, num_patch_sqrt: int = 16, device='cuda:0') -> None:
        '''
        :param data_paths:dataloader产生的一批数据路径
        :param label_paths: dataloader产生的一批标签路径
        :param rate: 各级模型之间的采样倍率
        :param num_patch: patch数量
        '''
        super().__init__()
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.rate = rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_focus = num_focus
        self.num_patch_sqrt = num_patch_sqrt

    @abstractmethod
    def getStageOneInputTensor(self) -> (Tensor, Tensor, Tensor):
        '''
        :return: 可直接用于训练训练的tensor，输入第一级别模型
        '''
        pass

    @abstractmethod
    def getStageTwoInputTensor(self, focus_index) -> (Tensor, Tensor, Tensor):
        '''
        :param focus_index: 由训练类计算得到的第一级高响应patch的索引
        :return: 可直接用于训练训练的tensor，输入第二级别模型
        '''
        pass

    @abstractmethod
    def getStageThreeInputTensor(self, focus_index1, focus_index2) -> (Tensor, Tensor):
        '''
        :param focus_index1: 由训练类计算得到的第一级高响应patch的索引
        :param focus_index2: 由训练类计算得到的第二级高响应patch的索引
        :return: 可直接用于训练训练的tensor，输入第二级别模型
        '''
        pass
