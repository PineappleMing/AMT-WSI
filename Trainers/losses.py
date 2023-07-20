import torch
import torch.nn as nn
import torch.nn.functional as F


def SLLoss(output, label, alpha=0.1, beta=1.0, reduction='mean', n_class=10):
    # label = torch.zeros(output.shape[0], n_class).cuda().scatter_(1, label.reshape(-1, 1), 1).float()
    if label.shape[-1] == 1 or len(label.shape) == 1:
        label = torch.eye(n_class)[label].cuda()
    label = label.float()
    if len(output.shape) == 1:
        output.unsqueeze(0)
        label.unsqueeze(0)
    A = -10

    y_true_1 = label.clone()
    y_pred_1 = (output >= 1e-7).float() * output + (output < 1e-7).float() * 1e-7
    y_true_2 = (label == 1).float() * label + (label == 0).float() * torch.e ** A
    y_pred_2 = output.clone()
    sl_loss = alpha * -torch.sum(y_true_1 * torch.log(y_pred_1), dim=-1) + beta * -torch.sum(
        y_pred_2 * torch.log(y_true_2), dim=-1)
    if reduction == 'mean':
        return sl_loss.mean()
    elif reduction == 'none':
        return sl_loss


def TLoss(output, label, alpha_target, alpha_prod, sublabel=None, smooth=0.0):
    label = label.float()
    if sublabel is None:
        sublabel = 1 - label
    # output = (output > 1 - 1e-4).float() * (1 - 1e-4) + ((output <= 1 - 1e-4) & (output >= 1e-7)).float() * output + (output < 1e-7).float() * 1e-7
    output = output * (1 - 1e-4) + 5e-5
    target_item = torch.prod(1 - output * label, dim=-1)
    prod_item = torch.prod(1 - output * sublabel, dim=-1)
    log_item = 1 - target_item ** alpha_target * prod_item ** alpha_prod
    tl = - torch.log(log_item)
    smooth_item = - torch.log(torch.sum(output * (1 - label), dim=-1))
    loss = tl * (1 - smooth) + smooth_item * smooth

    return loss


def TLoss_(output, label, alpha_target, alpha_prod, alpha_sub=0, sublabel=None, smooth=0.0):
    label = label.float()
    if sublabel is None:
        otherlabel = 1 - label
        sublabel = 1 - label - otherlabel
        alpha_prod = alpha_prod / (label.shape[-1] - 1)
    else:
        otherlabel = 1 - label - sublabel
        alpha_prod = alpha_prod / (label.shape[-1] - 2)
    # output = (output > 1 - 1e-4).float() * (1 - 1e-4) + ((output <= 1 - 1e-4) & (output >= 1e-7)).float() * output + (output < 1e-7).float() * 1e-7
    output = output * (1 - 1e-4) + 5e-5
    target_item = torch.prod(1 - output * label, dim=-1)
    sub_item = torch.prod(1 - output * sublabel, dim=-1)
    other_item = torch.prod(1 - output * otherlabel, dim=-1)
    log_item = 1 - target_item ** alpha_target * other_item ** alpha_prod * sub_item ** alpha_sub
    tl = - torch.log(log_item)
    smooth_item = - torch.log(torch.sum(output * (1 - label), dim=-1))
    loss = tl * (1 - smooth) + smooth_item * smooth

    return loss


def GradLoss(output, labels, target_layer, route_norm, mode='abs'):
    grad_loss = 0
    prediction = torch.sum(output * labels.float(), dim=1)
    for sample_idx in range(prediction.shape[0]):
        sample_loss = -torch.log(prediction[sample_idx])
        grads = torch.autograd.grad(outputs=sample_loss, inputs=target_layer.weight, retain_graph=True,
                                    create_graph=True, grad_outputs=torch.ones_like(sample_loss))
        grads = grads[0].abs().sum(dim=(2, 3))
        grad_loss += grads[route_norm[labels[sample_idx].argmax()] < 0.01].mean()
        # grad_loss += (grads-torch.tensor(route_norm[labels[sample_idx].argmax()]).cuda().float()).abs().mean()
    return grad_loss


def consistLoss(fea_c, label_c):
    N = label_c.shape[0]
    if N == 0:
        return torch.tensor([0.])
    if len(label_c.shape) == 2:
        label_c = torch.argmax(label_c, dim=-1)
    consist_mat = 1 - 2 * (label_c.unsqueeze(0) == label_c.unsqueeze(1)).float()
    fea_length = torch.sqrt(torch.sum(fea_c ** 2, dim=-1))
    cos_mat = torch.matmul(fea_c, fea_c.transpose(0, 1)) / (fea_length.unsqueeze(0) * fea_length.unsqueeze(1))
    consist_loss = cos_mat * consist_mat
    return consist_loss.mean()


def SmoothLoss(output, smooth_label):
    output = (output + 1e-4) / (1 + 1e-4)
    smooth_label = (smooth_label + 1e-4) / (1 + 1e-4)
    loss = -smooth_label * torch.log(output)
    reg = -smooth_label * torch.log(smooth_label)
    return loss.mean() + reg.mean() * 0.01


def GCELoss(output, label, q=0.9, weight=None, reduction='mean'):
    if len(label.shape) == 2:
        label = torch.argmax(label, dim=-1)
    if output.min() < 0 or output.max() > 1:
        output = nn.Softmax(dim=1)(output)
    pred = nn.NLLLoss(reduction='none')(-output, label)
    gce_loss = (1 - pred ** q) / q
    if not weight is None:
        weight = weight.to(gce_loss.device)
        gce_loss *= weight[label]
    if reduction == 'mean':
        return gce_loss.mean()
    elif reduction == 'none':
        return gce_loss


'''
# patch_labels/p_attn: (Batch, L*L)
def forward_attention_to_label(p_attn, patch_labels):
    p_attn = nn.Sigmoid()(p_attn)
    diff_mask = patch_labels.unsqueeze(1) - patch_labels.unsqueeze(2)
    score_mat = p_attn.unsqueeze(2) - p_attn.unsqueeze(1)
    attention_score = score_mat * diff_mask.float()
    attention_loss = attention_score[attention_score > 0].mean()
    return attention_loss
'''


# patch_labels/p_attn: (Batch, L*L)
def forward_attention(p_attn, g_attn, patch_labels, mask):
    p_attn = nn.Sigmoid()(p_attn)
    g_attn = g_attn.detach()
    patch_labels = patch_labels.float()
    g_max = g_attn.max(dim=1)[0].unsqueeze(1)
    g_min = g_attn.min(dim=1)[0].unsqueeze(1)
    g_attn = (g_attn - g_min) / (g_max - g_min)
    # attn_target = torch.sqrt(g_attn * patch_labels)
    attn_target = g_attn.masked_fill(~mask, 0)
    kl = attn_target * torch.log(attn_target / (p_attn + 1e-6) + 1e-6) + (1 - attn_target) * torch.log(
        (1 - attn_target) / (1 - p_attn + 1e-6) + 1e-6)
    return kl.mean()


# p_attn: (Batch, L*L) / (Batch*P, L*L)
# p_patch: (Batch*P) / (Batch*P*P)
# index: (Batch, P) / (Batch*P, P)
def backward_attention(p_attn, p_patch, index):
    N, P = index.shape
    p_attn = nn.Sigmoid()(p_attn)
    p_patch = p_patch.detach()
    p_attn = torch.cat([F.nll_loss(-p_attn, index[:, i], reduction='none').unsqueeze(1) for i in range(P)],
                       dim=1).flatten()
    kl = p_patch * torch.log(p_patch / (p_attn + 1e-6) + 1e-6) + (1 - p_patch) * torch.log(
        (1 - p_patch) / (1 - p_attn + 1e-6) + 1e-6)
    return kl.mean()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def a_g_loss(a_routes, g_routes, a_total, g_total, a_pred, a_weight=1, g_weight=10):
    L = len(a_routes)
    a_loss = sum([(a_total[i].max(dim=-1)[0].max(dim=-1)[0] -
                   a_routes[i][a_pred].unsqueeze(1).max(dim=-1)[0].max(dim=-1)[0]).abs().mean() for i in
                  range(L)]) * a_weight
    g_loss = sum([(g_total[i] - g_routes[i].unsqueeze(1)).abs().mean() for i in range(L)]) * g_weight
    ag_loss = a_loss + g_loss
    return ag_loss
