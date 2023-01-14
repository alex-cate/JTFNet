from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MPNCOV.python import MPNCOV


#
def make_model(args, parent=False):
    return MSG(args)


class MSG(nn.Module):
    def __init__(self, args):
        super(MSG, self).__init__()
        n_feats = args.n_feats
        self.scale = args.scale[0]
        act = nn.PReLU()

        # define head module
        self.color_head = nn.Sequential(
            common.ConvBlock(3, 49, kernel_size=7, stride=1, padding=3, bias=True, act=act),
            common.ConvBlock(49, n_feats, kernel_size=5, stride=1, padding=2, bias=True, act=act))
        self.depth_head = nn.Sequential(
            common.ConvBlock(1, 64, kernel_size=5, stride=1, padding=2, bias=True, act=act),
            common.DeconvBlock(64, n_feats, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True, act=act))

        self.sr_stage_x2 = nn.Sequential(
            common.ConvBlock(n_feats*2, n_feats, kernel_size=5, stride=1, padding=2, bias=True, act=act),
            common.ConvBlock(n_feats, n_feats, kernel_size=5, stride=1, padding=2, bias=True, act=act),
            common.ConvBlock(n_feats, n_feats, kernel_size=5, stride=1, padding=2, bias=True, act=act))

        self.depth_recon = nn.Sequential(common.ConvBlock(n_feats, 1, kernel_size=5, stride=1, padding=2, bias=True))

    def forward(self, depth, color):
        res = F.interpolate(depth, scale_factor=self.scale, mode='bicubic', align_corners=False)
        color = self.color_head(color)
        depth = self.depth_head(depth)

        depth_ten_x2 = torch.cat([depth, color], dim=1)
        depth_ten_x2 = self.sr_stage_x2(depth_ten_x2)

        depth_sr = self.depth_recon(depth_ten_x2)
        depth_sr = depth_sr + res
        return depth_sr
