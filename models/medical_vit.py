import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
# from models.module import *
from models.medical_hipt import get_vit256
from models.efficient_net import efficientnet_b0
from models.vision_transformer import vit_panda



class PatchEmbedding(nn.Module):
    def __init__(self, channel, patch_size1, patch_size2, dim, height =1600 , width=1200):
        super().__init__()
        # batch, channel, image_width, image_height = x.shape
        # assert image_width % patch_size == 0 and image_height % patch_size == 0, "Image size must be divided by the path size!"
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2

        num_patches_max = (height * width) // (patch_size1 * patch_size2)
        patch_dim = channel * patch_size1 * patch_size2

        self.to_patch_embedding = nn.Linear(patch_dim, dim, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches_max + 1, dim))

    def forward(self, x):
        b, c, w, h = x.shape
        x = rearrange(x, 'b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=self.patch_size1,
                      p2=self.patch_size2)  # 1, 3, 224, 224 = 1 ,3 ,(14, 16),(14, 16) => 1, 196, 768
        x1 = self.to_patch_embedding(x)

        num_patches = (h * w) // (self.patch_size1 * self.patch_size2)
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x2 = torch.cat([cls_token, x1], dim=1)
        x2 = x2 + self.pos_embedding[:,:(num_patches+1),:]

        return x2


class Attention(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)


    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=1)(attention_score)
        out = torch.bmm(attention_score, V)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, dropout=0.1, project_out=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout)
           )if project_out else nn.Identity()


    def forward(self, x):

        Q = rearrange(self.query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.key(x), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.value(x), 'b n (h d) -> b h n d', h=self.num_heads)
        attention_score = torch.einsum('b h q d, b h k d -> b h q k', Q, K)
        attention_score = nn.Softmax(dim=1)(attention_score) / math.sqrt(self.dim)
        out = torch.einsum('b h a n, b h n v -> b h a v', attention_score, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, dim=768, expansion=4, dropout=0.):

        super().__init__(
        nn.Linear(dim, expansion * dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * dim, dim),
        nn.Dropout(dropout)
        )

class FeedForwardBlock2(nn.Sequential):
    def __init__(self, dim=768, expansion=4, dropout=0.1):

        super().__init__(
        nn.Linear(dim, 50),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * dim, 10),
        nn.Dropout(dropout)
        )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class medical_former(nn.Module):
    def __init__(self,channel=3, patch_size=16, dim=384,num_heads=6, num_class=2):

        super().__init__()

        self.LN = nn.LayerNorm(dim)
        self.MHA = MultiHeadAttention(dim=dim,num_heads=num_heads)
        self.FFB = FeedForwardBlock(dim=dim)

        model256_path = '/home/lgj/medical_former/models/Checkpoints/vit256_small_dino.pth'
        self.model256 = get_vit256(pretrained_weights=model256_path, patch_size=patch_size)

        #self.efficient_net = efficientnet_b0()
        #self.PatchEmbedding = PatchEmbedding(channel=channel, patch_size1=patch_size1,patch_size2=patch_size2,dim=dim)

        patch_dim = channel * patch_size * patch_size

        self.embedding_to_class = Mlp(in_features=dim, hidden_features=dim//2,out_features=num_class)

    def forward(self, x):
        B, C, W, H = x.shape
        # encoder
        # [B, 257,384]
        x4, attn = self.model256(x)
        # decoder
        # # B, P ,E = x4.shape
        cls_token = x4[:, 0, :]
        x_patch = x4[:, 1:, :]
        x = torch.matmul(x_patch, cls_token.unsqueeze(2)).squeeze(2)   #(B, P)
        x_class = self.embedding_to_class(x_patch)
        x_class_all = self.embedding_to_class(cls_token)

        return x,  x_patch, x_class_all, attn


class medical_former_panda(nn.Module):
    def __init__(self, dim=384, num_class=5):

        super().__init__()

        self.model256 = vit_panda()
        self.embedding_to_class = Mlp(in_features=dim, hidden_features=dim//2,out_features=num_class)

    def forward(self, x):
        B, C, W, H = x.shape
        # encoder
        # [B, 257,384]
        x4, attn = self.model256(x)
        # decoder
        # # B, P ,E = x4.shape
        cls_token = x4[:, 0, :]
        x_patch = x4[:, 1:, :]
        x = torch.matmul(x_patch, cls_token.unsqueeze(2)).squeeze(2)   #(B, P)
        x_class = self.embedding_to_class(x_patch)
        x_class_all = self.embedding_to_class(cls_token)

        return x,  x_class, x_class_all, attn

class model_panda(nn.Module):
    def __init__(self, dim=384, num_class=5):
        super().__init__()
        self.model_one = medical_former_panda(dim=384, num_class=num_class)
        self.model_two = medical_former_panda(dim=384, num_class=num_class)
        self.model_three = medical_former_panda(dim=384, num_class=num_class)

    def forward(self, stage, x):
        if(stage == 0):
            return self.model_one(x)
        elif(stage == 1):
            #print(self.model_two.model256.patch_embed.proj.weight.max())
            return self.model_two(x)
        else:
            return self.model_three(x)


if __name__ == '__main__':
    import time
    start = time.time()
    device = torch.device('cpu')
    input = torch.randn(2,3,256,256).to(torch.device('cuda:0'))
    # model_trans = medical_former(channel=3, patch_size1=16, patch_size2=16, dim=384).to(torch.device('cuda:0'))

    # model256_path = './Checkpoints/vit256_small_dino.pth'
    # model256 = get_vit256(pretrained_weights=model256_path)
    model_vit_2 = vit_base_patch16_224_in21k(2, has_logits=False).to(torch.device('cuda:0'))
    # out = model256(input)
    # a,b,c,d = model_trans(input)
    out = model_vit_2(input)


    # patch_embedding6 = PatchEmbedding(input6,patch_size1=7,patch_size2=7,dim=100).to(device)
    # transformer6 = MvsformerEncoder().to(device)
    # transformer7 = MvsformerDecoder().to(device)
    # out = transformer6(input6)
    # out = transformer7(out)
    print("test")

