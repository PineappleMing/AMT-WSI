import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math

# 让channel数为8的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob  # 保留的比率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize  二值化
    # ---------------------------------------------------------------------------------------#
    #   对x先进行x.div(keep_prob)的放缩，这是为什么？
    #   回答：对权值放缩是为了获得输出的一致性，即期望不变。
    #       假设一个神经元的输出激活值为a，在不使用dropout的情况下，
    #       其输出期望值为a，如果使用了dropout，神经元就可能有保留和关闭两种状态，
    #       把它看作一个离散型随机变量，它就符合概率论中的0-1分布，
    #       其输出激活值的期望变为 p*a+(1-p)*0=pa，此时若要保持期望和不使用dropout时一致，就要除以p。
    # ---------------------------------------------------------------------------------------#
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,  # groups用来控制是使用DW卷积还是普通卷积
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # 又叫 Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),  # 使用BN，故bias为False
                                               norm_layer(out_planes),
                                               activation_layer())


# SE模块
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel，DW卷积channel不变
                 squeeze_factor: int = 4):  # 控制第一个FC层节点个数，等于 input_c // squeeze_factor
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)  # 1x1卷积 代替全连接层，作用相同
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # output_size=(1, 1)：对每个channel进行全局平均池化
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)  # 得到对应每个channel的对应程度
        return scale * x


# 对应每个MBConv模块的配置参数
class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5     两种可能
                 input_c: int,  # 输入MBConv的channel数
                 out_c: int,  # MBConv输出的channel数
                 expanded_ratio: int,  # 1 or 6     变胖倍数
                 stride: int,  # 1 or 2
                 use_se: bool,  # True       都用
                 drop_rate: float,  # MBConv中的随机失活比例
                 index: str,  # 1a, 2a, 2b, ...    用来记录当前MBConv模块的名称，方便后期分析
                 width_coefficient: float):  # 宽度倍率因子
        # 宽度*倍率因子，然后调整到离它最近的8的整数倍
        # 默认是EfficientNet_B0参数，B1~B7都是通过这样的倍率因子进行调节
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        #   变胖的channel数=输入channel数 * 变胖倍数
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


# MBConv模块
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,  # 传入配置文件
                 norm_layer: Callable[..., nn.Module]):  # BN
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 满足cnf.stride == 1 and cnf.input_c == cnf.out_c这两个条件，采用短连接
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        # 定义有序的字典，用于搭建MBConv结构
        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish   SiLU别名Swish

        # expand
        #   针对第一个MBConv，相等时跳过下面这个语句
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,  # DW卷积！
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})  # 线性激活层，不做任何处理

        self.block = nn.Sequential(layers)  # 把有序字典layers传给Sequential这个类
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1  # cnf.stride=1，则self.is_strided为False

        # 只有在使用shortcut连接 且drop_rate大于0时 才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)  # 通过 DropPath类 实现MBConv中的dropout
        else:
            self.dropout = nn.Identity()  # nn.Identity()表示不做任何处理

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,  # 宽度倍率因子，用于切换B0~B7
                 depth_coefficient: float,  # 深度倍率因子
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,  # 控制最后一个全连接层前的dropout层
                 drop_connect_rate: float = 0.2,  # 控制SE模块里的dropout
                 block: Optional[Callable[..., nn.Module]] = None,  # MBConv模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None  # BN
                 ):
        super(EfficientNet, self).__init__()

        # 默认配置表，stage2~stage8的配置参数
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        # 用于控制B0~B7中MBConv重复几次
        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))  # math.ceil(): 大于或等于，即向上取整

        # 默认是MBConv模块
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # partial()：往BN里传两个默认参数，下次再使用时可不用再传入这两个超参数了
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

            # partial()：往InvertedResidualConfig.adjust_channels里传参数
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0  # 用于记录MBConv模块序号
        # 遍历default_cnf，取每个list元素的最后一个参数repeats
        #   然后传给round_repeats函数，再求和
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        # 用于存储所有MBConv模块的配置文件
        inverted_residual_setting = []
        # stage是索引，args是数据
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            # 把每一行中的最后一个元素repeats pop出来进round_repeats，得到应该遍历这个MBConv模块的次数
            #   注意：此时cnf中没有repeats元素了
            for i in range(round_repeats(cnf.pop(-1))):
                # 针对重复多次的MBConv，非第一个block，有些参数要变一下
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides       此时cnf中没有repeats元素了
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                # MBConv中的droprate是逐步递增上去的
                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                # index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...     原来index长这样，权重里也是这个
                index = 'stage' + str(stage + 1) + chr(i + 97)  # stage1a, stage2a, stage2b, ...     修改一下，index长这样
                # 好好理一下 *cnf 操作
                # index用来记录当前MBConv模块的名称，方便后期分析
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers，通过有序字典进行构建
        #   往里面加东西，是update
        layers = OrderedDict()

        # first conv，命名为stem_conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     # out_planes针对B0是32，其它的不一定，需要用宽度倍率因子调整，
                                                     #   故用adjust_channels()函数进行操作
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            # 把index作为其名称，block就是上面写的MBConv类
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        # last_conv_input_c：最后那个1x1 conv输入channel是多少
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        # 起名叫top，也是人才
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        # 把有序字典layers传给nn.Sequential这个类，实例化得到self.features
        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:  # 在全连接层前添加dropout层？是的，没看到
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        # 把classifier列表中的元素通过 *list 依次传给nn.Sequential这个类，实例化得到self.classifier
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# 通过上面的EfficientNet类得到efficientnet_b0 ~ efficientnet_b7
def efficientnet_b0(num_classes=2):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)

if __name__ == '__main__':

    model = efficientnet_b0()
    input = torch.randn(4,3,256,256)

    out = model(input)
    print("test")