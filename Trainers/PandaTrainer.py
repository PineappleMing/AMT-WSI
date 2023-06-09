# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from Analyzers import CommonAnalyzer
from dataloader_adapters import CamelonInputTensorAdapter, CamelonTestTensorAdapter
from dataloader_adapters.Panda.PandaInputTensorAdapter import PandaInputTensorAdapter
from models import *
from dataset import *
from torch.utils.data import Dataset, DataLoader  # 数据包
from models.medical_vit import Mlp
import HP as HP
from losses import *

params = {}  # 初始参数设计
params['num_epoch'] = 100  # 训练的轮数
params['batch_size'] = 4  # 一批次进入的数据大小
params['lr'] = [0.002, 0.002, 0.01]  # 学习率
params['lr_attn'] = [0.01, 0.01]
params['square_size'] = 256
params['patch_size'] = 32
params['num_patch'] = 64
params['num_patch_sqrt'] = 8
params['gap_size'] = 8  # transformer等级之间间隔的倍数
params['num_focus'] = 4  # 取前多少个框
params['num_class'] = 2

train_dataset = Data_Panda()
test_dataset = Data_Panda_Test()

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=params['batch_size'])
test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=params['batch_size'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model1 = medical_former(channel=3, patch_size=params['patch_size'], dim=384, num_heads=10).to(device)
model2 = medical_former(channel=3, patch_size=params['patch_size'], dim=384, num_heads=10).to(device)
model3 = medical_former(channel=3, patch_size=params['patch_size'], dim=384, num_heads=10).to(device)
attn_clsfr1 = Mlp(in_features=384, hidden_features=192, out_features=1).to(device)
attn_clsfr2 = Mlp(in_features=384, hidden_features=192, out_features=1).to(device)
# model1.load_state_dict(torch.load(HP.models_path + 'panda1.pth'))
# model2.load_state_dict(torch.load(HP.models_path + 'panda2.pth'))
# model3.load_state_dict(torch.load(HP.models_path + 'panda3.pth'))
# attn_clsfr1.load_state_dict(torch.load(HP.models_path + 'panda_atten1.pth'))
# attn_clsfr2.load_state_dict(torch.load(HP.models_path + 'panda_atten2.pth'))

opt_cls = torch.optim.SGD([{'params': model1.parameters(), 'lr': params['lr'][0]},
                           {'params': model2.parameters(), 'lr': params['lr'][1]},
                           {'params': model3.parameters(), 'lr': params['lr'][2]}])
opt_attn = torch.optim.SGD([{'params': attn_clsfr1.parameters(), 'lr': params['lr_attn'][0]},
                            {'params': attn_clsfr2.parameters(), 'lr': params['lr_attn'][1]}])
loss_se = HP.loss_schedule(max_epoch=20)

writer = SummaryWriter(log_dir=HP.log_dir + '/panda/')


def save_focus_map(score, pth, norm=True, target_size=256):
    N = score.shape[0]
    N_sqrt = int(np.sqrt(N))
    score = nn.Sigmoid()(score).detach().cpu().numpy()
    if norm:
        score = (score - score.min()) / (score.max() - score.min())
    score = (score * 255).astype(np.uint8)
    focus_map = Image.fromarray(score.reshape(N_sqrt, N_sqrt)).resize((target_size, target_size))
    focus_map.save(pth)


def train(epoch):
    analyzer = CommonAnalyzer(writer)
    model1.train()
    model2.train()
    model3.train()
    loss_schedule = loss_se[epoch]
    weight1 = torch.tensor([0.6, 1.0])
    weight2 = torch.tensor([0.2, 1.0])
    weight3 = torch.tensor([0.2, 1.0])

    for train_i, (medical_tag_path, label_path) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + train_i
        input_adapter = PandaInputTensorAdapter(data_paths=medical_tag_path, label_paths=label_path,
                                                rate=params['gap_size'], num_focus=params['num_focus'],
                                                num_patch_sqrt=params['num_patch_sqrt'])

        # ------------------------------------------------------------------------------------------------
        inputs1, stage_one_label, stage_one_patch_label, stage_one_true_label, stage_one_true_patch_label = input_adapter.getStageOneInputTensor()
        stage_one_out, stage_one_fea, stage_one_class, attn = model1(inputs1)
        B, _ = stage_one_out.shape
        stage_one_attention = attn_clsfr1(stage_one_fea.detach()).squeeze(2)
        # for i in range(params['num_focus']):
        #     slide_name = medical_tag_path[i].split('/')[-1].split('.')[0]
        #     save_focus_map(stage_one_attention[i].cpu(), "/home/lhm/tmp/panda_test_focus/" + slide_name + '.png')
        g_attn1 = torch.autograd.grad(outputs=nn.Softmax(dim=1)(stage_one_class)[:, 1].sum(),
                                      inputs=attn,
                                      retain_graph=True)[0]
        g_attn1 = g_attn1.sum(dim=(1, 2))[:, 1:]
        value, stage_one_index = torch.sort(stage_one_attention, 1, descending=False)
        stage_one_fine_index = stage_one_index[:, -params['num_focus']:].detach()
        forward_attention_loss1 = forward_attention(stage_one_attention, g_attn1, stage_one_label)
        criterion1 = nn.CrossEntropyLoss(weight=weight1).to(device)
        loss_stage_one_all = criterion1(stage_one_class, stage_one_label)
        analyzer.updateStageOne(stage_one_true_label, stage_one_class, stage_one_true_patch_label, stage_one_fine_index)
        weight1 = torch.clamp(torch.tensor([analyzer.label_rate[0] / (1.01 - analyzer.label_rate[0]), 1.]), min=0.1,
                              max=1.0)

        # ------------------------------------------------------------------------------------------------
        if loss_schedule[1][0] > 0:
            inputs2, stage_two_label, stage_two_patch_label, stage_two_true_label, stage_two_true_patch_label = input_adapter.getStageTwoInputTensor(
                stage_one_fine_index)
            stage_two_out, stage_two_fea, stage_two_class, attn = model2(inputs2)
            stage_two_attention = attn_clsfr2(stage_two_fea.detach()).squeeze(2)
            g_attn2 = torch.autograd.grad(outputs=nn.Softmax(dim=1)(stage_two_class)[:, 1].sum(),
                                          inputs=attn,
                                          retain_graph=True)[0]
            g_attn2 = g_attn2.sum(dim=(1, 2))[:, 1:]
            value, stage_two_index = torch.sort(stage_two_attention, 1, descending=False)
            stage_two_fine_index = stage_two_index[:, -params['num_focus']:].detach()
            forward_attention_loss2 = forward_attention(stage_two_attention, g_attn2, stage_two_label)
            criterion2 = nn.CrossEntropyLoss(weight=weight2).to(device)
            stage_two_patch_label = stage_two_patch_label.reshape(B * params['num_focus'], -1)
            stage_two_true_patch_label = stage_two_true_patch_label.reshape(B * params['num_focus'], -1)
            stage_two_label = stage_two_label.reshape(B * params['num_focus'])
            stage_two_true_label = stage_two_true_label.reshape(B * params['num_focus'])
            stage_two_class = stage_two_class.reshape(B * params['num_focus'], 2)
            loss_stage_two_all = criterion2(stage_two_class, stage_two_label)
            loss_stage_two_all = GCELoss(stage_two_class, stage_two_label, weight=weight2)
            analyzer.updateStageTwo(stage_two_true_label, stage_two_class, stage_two_true_patch_label,
                                    stage_two_fine_index)
            weight2 = torch.clamp(torch.tensor([analyzer.label_rate[1] / (1.01 - analyzer.label_rate[1]), 1.]), min=0.1,
                                  max=1.0)

        # ------------------------------------------------------------------------------------------------
        if loss_schedule[2][0] > 0:
            inputs3, stage_three_label, stage_three_true_label = input_adapter.getStageThreeInputTensor(
                stage_one_fine_index,
                stage_two_fine_index)
            stage_three_out, stage_three_fea, stage_three_class, attn = model3(inputs3)
            stage_three_label = stage_three_label.reshape(B * params['num_focus'] * params['num_focus'])
            stage_three_true_label = stage_three_true_label.reshape(B * params['num_focus'] * params['num_focus'])
            stage_three_class = stage_three_class.reshape(B * params['num_focus'] * params['num_focus'], 2)
            criterion3 = nn.CrossEntropyLoss(weight=weight3).to(device)
            loss_stage_three_all = criterion3(stage_three_class, stage_three_label)
            # loss_stage_three_all = GCELoss(stage_three_class, stage_three_label, weight=weight3)
            analyzer.updateStageThree(stage_three_true_label, stage_three_class)
            weight3 = torch.clamp(torch.tensor([analyzer.label_rate[2] / (1.01 - analyzer.label_rate[2]), 1.]), min=0.1,
                                  max=1.0)

        # ------------------------------------------------------------------------------------------------
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        # stage1 loss
        distill_loss_1 = 0
        backward_attention_loss1 = 0
        if loss_schedule[0][2] > 0:
            stage_2_back = torch.mean(stage_two_class.reshape((-1, params['num_focus'], 2)), dim=1).detach()
            distill_loss_1 = kl_loss(F.log_softmax(stage_one_class / HP.temp, dim=1),
                                     F.softmax(stage_2_back / HP.temp, dim=1))
        if loss_schedule[0][3] > 0:
            stage_2_cls_prob = F.softmax(stage_two_class, dim=1)[:, 1].detach()
            backward_attention_loss1 = backward_attention(stage_one_attention, stage_2_cls_prob,
                                                          stage_one_fine_index)
        loss_cls = loss_schedule[0][0] * loss_stage_one_all + loss_schedule[0][2] * distill_loss_1
        loss_attn = loss_schedule[0][1] * forward_attention_loss1 + loss_schedule[0][
            3] * backward_attention_loss1

        # stage2 loss
        distill_loss_2 = 0
        backward_attention_loss2 = 0
        if loss_schedule[1][0] > 0:
            if loss_schedule[1][2] > 0:
                stage_3_back = torch.mean(stage_three_class.reshape((-1, params['num_focus'], 2)), dim=1).detach()
                distill_loss_2 = kl_loss(F.log_softmax(stage_two_class / HP.temp, dim=1),
                                         F.softmax(stage_3_back / HP.temp, dim=1))
            if loss_schedule[1][3] > 0:
                stage_3_cls_prob = F.softmax(stage_three_class, dim=1)[:, 1].detach()
                backward_attention_loss2 = backward_attention(stage_two_attention, stage_3_cls_prob,
                                                              stage_two_fine_index)
            loss_cls += loss_schedule[1][0] * loss_stage_two_all + loss_schedule[1][2] * distill_loss_2
            loss_attn += loss_schedule[1][1] * forward_attention_loss2 + loss_schedule[1][
                3] * backward_attention_loss2

        # stage3 loss
        if loss_schedule[2][0] > 0:
            loss_cls += loss_stage_three_all * loss_schedule[2][0]

        opt_cls.zero_grad()
        loss_cls.backward()
        opt_cls.step()
        opt_attn.zero_grad()
        loss_attn.backward()
        opt_attn.step()

        analyzer.print(epoch, loss_se[epoch])
        analyzer.saveToFile(step=global_step, mode='train')
        if global_step > 0 and global_step % 200 == 0:
            torch.save(model1.state_dict(), HP.models_path + 'panda_gce1.pth')
            torch.save(model2.state_dict(), HP.models_path + 'panda_gce2.pth')
            torch.save(model3.state_dict(), HP.models_path + 'panda_gce3.pth')
            torch.save(attn_clsfr1.state_dict(), HP.models_path + 'panda_atten_gce1.pth')
            torch.save(attn_clsfr2.state_dict(), HP.models_path + 'panda_atten_gce2.pth')


def test(epoch):
    model1.eval()
    model2.eval()
    model3.eval()
    analyzer = CommonAnalyzer(writer)
    for train_i, (medical_tag_path, label_path) in enumerate(test_loader):
        input_adapter = PandaInputTensorAdapter(data_paths=medical_tag_path, label_paths=label_path,
                                                rate=params['gap_size'], num_focus=params['num_focus'])
        # ------------------------------------------------------------------------------------------------
        inputs1, stage_one_label, stage_one_patch_label, stage_one_true_label, stage_one_true_patch_label = input_adapter.getStageOneInputTensor()
        stage_one_out, stage_one_fea, stage_one_class, attn = model1(inputs1)
        B, _ = stage_one_out.shape
        stage_one_attention = attn_clsfr1(stage_one_fea.detach()).squeeze(2)
        value, stage_one_index = torch.sort(stage_one_attention, 1, descending=False)
        stage_one_fine_index = stage_one_index[:, -params['num_focus']:].detach()
        analyzer.updateStageOne(stage_one_true_label, stage_one_class, stage_one_true_patch_label, stage_one_fine_index)

        # ------------------------------------------------------------------------------------------------
        inputs2, stage_two_label, stage_two_patch_label, stage_two_true_label, stage_two_true_patch_label = input_adapter.getStageTwoInputTensor(
            stage_one_fine_index)
        stage_two_out, stage_two_fea, stage_two_class, attn = model2(inputs2)
        stage_two_attention = attn_clsfr2(stage_two_fea.detach()).squeeze(2)
        value, stage_two_index = torch.sort(stage_two_attention, 1, descending=False)
        stage_two_fine_index = stage_two_index[:, -params['num_focus']:].detach()
        stage_two_patch_label = stage_two_patch_label.reshape(B * params['num_focus'], -1)
        stage_two_label = stage_two_label.reshape(B * params['num_focus'])
        stage_two_class = stage_two_class.reshape(B * params['num_focus'], 2)
        analyzer.updateStageTwo(stage_two_true_label, stage_two_class, stage_two_true_patch_label, stage_two_fine_index)

        # ------------------------------------------------------------------------------------------------
        inputs3, stage_three_label, stage_three_true_label = input_adapter.getStageThreeInputTensor(
            stage_one_fine_index,
            stage_two_fine_index)
        stage_three_out, stage_three_fea, stage_three_class, attn = model3(inputs3)
        stage_three_label = stage_three_label.reshape(B * params['num_focus'] * params['num_focus'])
        stage_three_true_label = stage_three_true_label.reshape(B * params['num_focus'] * params['num_focus'])
        stage_three_class = stage_three_class.reshape(B * params['num_focus'] * params['num_focus'], 2)
        analyzer.updateStageThree(stage_three_true_label, stage_three_class)
        analyzer.print(epoch, loss_se[epoch])


for epoch in range(params['num_epoch']):
    train(epoch)
    # test(epoch)
