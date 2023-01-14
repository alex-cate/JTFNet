from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MPNCOV.python import MPNCOV


#
def make_model(args, parent=False):
    return DepthNetX8(args)


## Kernel Generation Function(KG)
class KG(nn.Module):
    def __init__(self, type, n_feats, filter_size, group_num, bias=True, act=nn.PReLU()):
        super(KG, self).__init__()

        self.conv1 = nn.Sequential(common.ConvBlock(n_feats, n_feats, kernel_size=5, stride=1, padding=2, bias=bias,
                                                    act=act))
        if type == "down":
            self.conv2 = nn.Sequential(
                common.ConvBlock(n_feats, n_feats, kernel_size=5, stride=2, padding=2, bias=bias, act=act))

        elif type == "up":
            self.conv2 = nn.Sequential(
                common.DeconvBlock(n_feats, n_feats, kernel_size=5, stride=2, padding=2, output_padding=1, bias=bias,
                                   act=act))
        else:
            self.conv2 = nn.Sequential(
                common.ConvBlock(n_feats, n_feats, kernel_size=5, stride=1, padding=2, bias=bias, act=act))
        self.conv3 = nn.Sequential(
            common.ConvBlock(n_feats, filter_size * filter_size * group_num, kernel_size=1, stride=1, padding=0,
                             bias=bias))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return y


## Joint Triliteral Filter (JTF)
class JBF(nn.Module):
    def __init__(self, type, n_feats, kernel_size, padding, filter_size, group_num, bias=True, act=nn.PReLU()):
        super(JBF, self).__init__()
        self.filter_size = filter_size
        self.g = group_num
        self.bottleneck = nn.Sequential(common.ConvBlock(n_feats, 32, kernel_size=1, stride=1, padding=0, bias=bias,
                                                         act=act))
        self.unfold = nn.Unfold(filter_size, 1, (filter_size - 1) // 2, 1)
        self.target_kg = KG(type="general", n_feats=n_feats, filter_size=filter_size, group_num=group_num, bias=True,
                            act=act)
        self.guidance_kg = KG(type=type, n_feats=n_feats, filter_size=filter_size, group_num=group_num, bias=True,
                              act=act)
        self.jbf_conv = nn.Sequential(
            common.ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=bias,
                             act=act),
            common.ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=bias))

    def forward(self, source, guidance):
        residual = source
        b, c, h, w = source.shape
        residual = self.target_kg(residual)
        guidance = self.guidance_kg(guidance)
        bi_kernel = (residual * guidance).view(b, self.g, self.filter_size * self.filter_size, h, w).unsqueeze(2)
        patch = self.unfold(source).view(b, c, self.filter_size * self.filter_size, h, w)
        patch = patch.view(b, self.g, c // self.g, self.filter_size * self.filter_size, h, w)
        jbf_new = (patch * bi_kernel).view(b, c, self.filter_size * self.filter_size, h, w).sum(dim=2)
        # bi_kernel = residual * guidance        
        # patch = self.unfold(source).view(b, c * self.filter_size * self.filter_size, h, w)
        # jbf_new = (patch * bi_kernel).view(b, self.filter_size * self.filter_size, c, h, w).sum(dim=1)
        jbf_new = self.jbf_conv(jbf_new)
        source = jbf_new + source

        return source


## Multi-scale Fusion Module
class Multi_Scale_Fusion(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, filter_size, group_num, bias=True, act=nn.PReLU()):
        super(Multi_Scale_Fusion, self).__init__()
        self.low_jtf = JBF(type="down", n_feats=n_feats, kernel_size=kernel_size, padding=padding,
                           filter_size=filter_size, group_num=group_num, bias=bias, act=act)
        self.high_jtf = JBF(type="up", n_feats=n_feats, kernel_size=kernel_size, padding=padding,
                            filter_size=filter_size, group_num=group_num, bias=bias, act=act)
        self.update = JBF(type="up", n_feats=n_feats, kernel_size=kernel_size, padding=padding,
                          filter_size=filter_size, group_num=group_num, bias=bias, act=act)

    def forward(self, low, high):
        low_n = self.low_jtf(low, high)
        high_n = self.high_jtf(high, low)
        high_n = self.update(high_n, low_n)

        return high_n


## Invertion Fusion Module
class Invertion_Fusion(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, filter_size, group_num, bias=True, act=nn.PReLU()):
        super(Invertion_Fusion, self).__init__()
        self.low_jtf = JBF(type="down", n_feats=n_feats, kernel_size=kernel_size, padding=padding,
                           filter_size=filter_size, group_num=group_num, bias=bias, act=act)
        self.high_jtf = JBF(type="up", n_feats=n_feats, kernel_size=kernel_size, padding=padding,
                            filter_size=filter_size, group_num=group_num, bias=bias, act=act)
        self.update = JBF(type="down", n_feats=n_feats, kernel_size=kernel_size, padding=padding,
                          filter_size=filter_size, group_num=group_num, bias=bias, act=act)

    def forward(self, low, high):
        low_n = self.low_jtf(low, high)
        high_n = self.high_jtf(high, low)
        low_n = self.update(low_n, high_n)

        return low_n


## Color Image Function
class Color_Function(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, bias=True, act=nn.PReLU()):
        super(Color_Function, self).__init__()
        self.feature = nn.Sequential(
            common.ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=bias,
                             act=act))
        self.down = nn.Sequential(common.MaxpoolingBlock(kernel_size=kernel_size, stride=2, padding=padding))

    def forward(self, color):
        color = self.feature(color)
        color = self.down(color)

        return color


## Super Resolution stage of one scale
class SR_Stage(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, filter_size, group_num, bias=True, act=nn.PReLU()):
        super(SR_Stage, self).__init__()
        G0 = 32
        G = n_feats
        C = 8
        self.rdbs = nn.Sequential(common.RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.jbf1 = JBF(type="general", n_feats=n_feats, kernel_size=kernel_size, group_num=group_num, padding=padding,
                        filter_size=filter_size, bias=True, act=act)
        self.jbf2 = JBF(type="general", n_feats=n_feats, kernel_size=kernel_size, group_num=group_num, padding=padding,
                        filter_size=filter_size, bias=True, act=act)

        self.tensor = nn.Sequential(
            common.ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=bias))
        self.up = nn.Sequential(
            common.DeconvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1,
                               bias=bias, act=act))

    def forward(self, depth, color):
        depth_tensor = self.rdbs(depth)
        depth_tensor = self.jbf1(depth_tensor, color)
        depth_tensor = self.jbf2(depth_tensor, color)
        depth_tensor = self.tensor(depth_tensor)
        depth = depth + depth_tensor
        depth = self.up(depth)

        return depth


## CrossEdge_Net (CEN)
class DepthNetX8(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DepthNetX8, self).__init__()
        n_feats = args.n_feats
        act = nn.PReLU()
        filter_size = 7
        kernel_size = 5
        padding = 2
        group_num = 8
        self.scale = args.scale[0]

        # define head module
        self.color_head = nn.Sequential(
            common.ConvBlock(3, n_feats, kernel_size=7, stride=1, padding=3, bias=True, act=act),
            common.ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True, act=act))
        self.depth_head = nn.Sequential(
            common.ConvBlock(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True, act=act),
            common.ConvBlock(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True, act=act))

        self.color_fun = nn.ModuleList()
        self.sr_stage = nn.ModuleList()
        self.multiscale_fusion = nn.ModuleList()
        for i in range(4):
            self.sr_stage.append(SR_Stage(n_feats, kernel_size, padding, filter_size, group_num, bias=True, act=act))
        for i in range(3):
            self.color_fun.append(Color_Function(n_feats, kernel_size, padding, bias=True, act=act))
        for i in range(2):
            self.multiscale_fusion.append(
                Multi_Scale_Fusion(n_feats, kernel_size, padding, filter_size, group_num, bias=True, act=act))
        self.invertion_fusion = Invertion_Fusion(n_feats, kernel_size, padding, filter_size, group_num, bias=True,
                                                 act=act)
        self.depth_recon = nn.Sequential(
            common.ConvBlock(n_feats, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True))

    def forward(self, depth, color):
        res = F.interpolate(depth, scale_factor=self.scale, mode='bicubic', align_corners=False)
        depth = self.depth_head(depth)
        color = self.color_head(color)

        # color_ten_x16 = self.color_fun[0](color)
        # color_ten_x8 = self.color_fun[1](color_ten_x16)
        # color_ten_x4 = self.color_fun[2](color_ten_x8)
        # color_ten_x2 = self.color_fun[3](color_ten_x4)

        color_ten_x8 = self.color_fun[0](color)
        color_ten_x4 = self.color_fun[1](color_ten_x8)
        color_ten_x2 = self.color_fun[2](color_ten_x4)

        depth_ten_x2 = self.sr_stage[0](depth, color_ten_x2)

        depth_ten_x4 = self.sr_stage[1](depth_ten_x2, color_ten_x4)
        update_x4 = self.multiscale_fusion[0](depth_ten_x2, depth_ten_x4)

        depth_ten_x8 = self.sr_stage[2](update_x4, color_ten_x8)
        update_x8 = self.multiscale_fusion[1](update_x4, depth_ten_x8)

        depth_ten_x16 = self.sr_stage[3](update_x8, color)
        update_x8 = self.invertion_fusion(update_x8, depth_ten_x16)

        depth_sr = self.depth_recon(update_x8)
        depth_sr = depth_sr + res

        return depth_sr
