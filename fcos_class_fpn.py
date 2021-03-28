# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps. 在特征图列表上添加FPN
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive 假设特征图在列表中是按深度连续递增排列
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []######一个列表
        self.layer_blocks = []######一个列表

###########构造nn.quential类，有名字

        for idx, in_channels in enumerate(in_channels_list, 1):  # 从1开始计idx
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            # 注意这里layer_block_module输出通道数都是out_channels，比如256
            # 也就是说fpn每一层级的特征图输出通道数是一样的
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        具体每层流向可以对照着看我下面一幅图
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
            C系列，i.e. [C3, C4, C5] 其实就是resnet(body)的输出
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
            P系列，i.e. [P3, P4, P5, P6, P7]
        """

############倒序进行FPN 所谓topdown
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
############最上层得到Pi后，再经过一个3*3网络
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))


        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
############[:-1]列表从0到倒数第二个数,[::-1]倒序
            if not inner_block:
                continue
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")



            inner_lateral = getattr(self, inner_block)(feature)
######Ci到Pi的卷积



            inner_top_down = F.interpolate(
                last_inner, size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='nearest'
            )
######上采样 input=last_inner, size=(H,W)=(.shap[-2],.shape[-1])(torch中核的shape应该是（c,h,w）)


            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

#######输出是Pi的卷积后的结果
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        # 在C5或者P5的基础上再来两层卷积得到P6,P7
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]




"""
----------------------------------------------------------------------------
FPN 结构：
有Ci Pi Oi
Ci经过一个1*1conv改变channel 为outpchannel 并且和Pi+1 上采样相加得到Pi
Pi经过一个不改变channel的3*3conv得到Oi
P7,P6没有Ci与之对应，可通过maxpooling或者conv降采样得到

"""