# -*- coding: utf-8 -*-
import torch
from torch import nn

from Analyzers import CommonAnalyzer
from Trainers import HP
from dataloader_adapters import CamelonInputTensorAdapter
from models import *
from dataset import *
from torch.utils.data import Dataset, DataLoader  # 数据包
from models.medical_vit import Mlp


class AMT(nn.Module):
    def __init__(self, patch_size, num_focus) -> None:
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model1 = medical_former(channel=3, patch_size=16, dim=384, num_heads=10).to(self.device)
        self.model2 = medical_former(channel=3, patch_size=16, dim=384, num_heads=10).to(self.device)
        self.model3 = medical_former(channel=3, patch_size=16, dim=384, num_heads=10).to(self.device)
        self.attn_clsfr1 = Mlp(in_features=384, hidden_features=192, out_features=1).to(self.device)
        self.attn_clsfr2 = Mlp(in_features=384, hidden_features=192, out_features=1).to(self.device)
        # change to /path/to/models
        self.model1.load_state_dict(torch.load(HP.models_path + 'CA_camel1.pth'))
        self.model2.load_state_dict(torch.load(HP.models_path + 'CA_camel2.pth'))
        self.model3.load_state_dict(torch.load(HP.models_path + 'CA_camel3.pth'))
        self.attn_clsfr1.load_state_dict(torch.load(HP.models_path + 'CA_camel_atten1.pth'))
        self.attn_clsfr2.load_state_dict(torch.load(HP.models_path + 'CA_camel_atten2.pth'))
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        analyzer = CommonAnalyzer(log_dir=HP.log_dir)

    def forward(self, x):
        pass
