import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, padding=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=padding, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class ConvBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=2, bias=False, act=None):
        super(ConvBlock, self).__init__()

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, stride=stride, bias=bias)
        ]
        if act is not None: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class DeconvBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=2, output_padding=0, bias=False, act=None):
        super(DeconvBlock, self).__init__()
        m = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding,
                                output_padding=output_padding, bias=bias)]
        if act is not None: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class MaxpoolingBlock(torch.nn.Module):
    def __init__(self, kernel_size, stride=2, padding=1, act=None):
        super(MaxpoolingBlock, self).__init__()
        m = [nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)]
        if act is not None: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class AvgpoolingBlock(torch.nn.Module):
    def __init__(self, kernel_size, stride=2, padding=1, act=None):
        super(AvgpoolingBlock, self).__init__()
        m = [nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)]
        if act is not None: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

# class Upsampler(nn.Sequential):
#     def __init__(self, n_feats, bias=True):
#         super(Upsampler, self).__init__()
#         self.conv=ConvBlock(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.pixelshufle=nn.PixelShuffle(2)
# 
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pixelshufle(x)
#         return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                # m.append(conv(n_feat, 2 * n_feat, 3, bias))
                # m.append(conv(2 * n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x