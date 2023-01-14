from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MPNCOV.python import MPNCOV


#
def make_model(args, parent=False):
    return CEN(args)


## Kernel_generation (KG)
class KG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, act=nn.PReLU()):
        super(KG, self).__init__()

        # self.conv_first = nn.Sequential(conv(n_feat, n_feat, 5, padding=2, bias=bias), act,
        #                                 conv(n_feat, n_feat, 5, padding=2, bias=bias), act,
        #                                 conv(n_feat, n_feat, 5, padding=2, bias=bias), act,
        #                                 conv(n_feat, n_feat, 5, padding=2, bias=bias)
        #                                 )
        # self.kernel_gen = conv(n_feat, kernel_size * kernel_size * n_feat, 1, padding=0, bias=bias)

        self.conv_first = nn.Sequential(conv(n_feat, n_feat, 3, padding=1, bias=bias), act,
                                        conv(n_feat, n_feat, 3, padding=1, bias=bias), act,
                                        conv(n_feat, n_feat, 3, padding=1, bias=bias), act,
                                        conv(n_feat, n_feat, 3, padding=1, bias=bias), act,
                                        conv(n_feat, kernel_size * kernel_size * n_feat, 1, padding=0, bias=bias))

        self.kernel = nn.Tanh()

    def forward(self, x):
        y = self.conv_first(x)
        # y = x + y
        # y = self.kernel_gen(y)
        # y=self.kernel(y)
        return y


## Joint Biliteral Filter (JBF)
class JBF(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act=nn.PReLU()):
        super(JBF, self).__init__()
        self.kernel_size = kernel_size
        self.tensor = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size=1, padding=0), act)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, 1)
        self.target_kg = KG(conv, n_feat, kernel_size, bias=True, act=nn.PReLU())
        self.guidance_kg = KG(conv, n_feat, kernel_size, bias=True, act=nn.PReLU())
        self.concat_kg = KG(conv, n_feat, kernel_size, bias=True, act=nn.PReLU())
        self.jbf_conv = nn.Sequential(conv(n_feat, n_feat, 3, padding=1), act, conv(n_feat, n_feat, 3, padding=1))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, image, guidance):
        # image, guidance = x
        residual = image
        # batch_size1, c1, h1, w1 = image.shape
        # batch_size2, c2, h2, w2 = guidance.shape
        # print(c1,h1,w1)
        # print(c2,h2,w2)
        ten_cat = torch.cat((image, guidance), 1)
        ten_cat = self.tensor(ten_cat)
        bi_kernel = self.concat_kg(ten_cat)
        # residual = self.target_kg(residual)
        # guidance = self.guidance_kg(guidance)
        # residual = self.concat_kg(residual)
        # guidance = self.concat_kg(guidance)
        # bi_kernel = residual * guidance
        b, c, h, w = image.shape
        patch = self.unfold(image).view(b, c * self.kernel_size * self.kernel_size, h, w)
        jbf_new = (patch * bi_kernel).view(b, self.kernel_size * self.kernel_size, c, h, w).sum(dim=1)
        jbf_new = self.jbf_conv(jbf_new)
        image = jbf_new + image
        return image


## Edge Guidance Extraction Module (EGEM)
class EGEM(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, tensor_num, bias=True, act=nn.PReLU()):
        super(EGEM, self).__init__()
        # print(tensor_num)
        self.tensor = nn.Sequential(conv(n_feat * tensor_num, n_feat, kernel_size=1, padding=0, bias=bias), act)
        self.jbf = JBF(conv, 32, kernel_size, act=nn.PReLU())
        self.rb1 = common.ResBlock(conv, n_feat, 3)
        self.rb2 = common.ResBlock(conv, n_feat, 3)

    def forward(self, edge, feature):
        # edge, feature = x
        feature = self.tensor(feature)
        edge = self.jbf(edge, feature)
        edge = self.rb1(edge)
        edge = self.rb2(edge)

        return edge


## CrossEdge_Net (CEN)
class CEN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CEN, self).__init__()
        n_groups = args.n_groups
        n_feats = args.n_feats
        kernel_size = 7
        scale = args.scale[0]
        act = nn.PReLU()
        self.n_groups = n_groups

        if scale == 2:
            filter_size = 6
        # elif scale == 4:
        #     filter_size = 8
        else:
            filter_size = 8

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean1 = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.sub_mean2 = common.MeanShift(1, rgb_mean, rgb_std)

        # define head module
        modules_head_image = [conv(args.n_colors, 128, 3, padding=1), act,
                              conv(128, n_feats, 3, padding=1), act]
        modules_head_edge = [conv(args.n_colors, 128, 3, padding=1), act,
                             conv(128, n_feats, 3, padding=1), act]
        self.tensor = nn.Sequential(conv(n_feats, n_feats, 1, padding=0, bias=True), act)
        self.rb = common.ResBlock(conv, n_feats, 3, padding=1)
        self.egems = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(n_groups):
            self.egems.append(EGEM(conv, n_feats, kernel_size, i + 1, bias=True, act=act))
            # self.egems.append(EGEM(conv, n_feats, kernel_size, 1, bias=True, act=act))
            self.ups.append(common.UpBlock(n_feats, kernel_size=filter_size, stride=scale, padding=2, bias=True))
            self.downs.append(common.DownBlock(n_feats, kernel_size=filter_size, stride=scale, padding=2, bias=True))
        # self.deconv=common.DeconvBlock(n_feats, n_feats, 3, stride=scale, bias=False, act=act)
        self.image_up = common.UpBlock(n_feats, kernel_size=filter_size, stride=scale, padding=2, bias=True)
        self.edge_up = common.Upsampler(conv, scale, n_feats, act=False)
        self.fusion = conv(n_feats * (n_groups + 1), n_feats, 1, padding=0)
        self.jbf = JBF(conv, 32, kernel_size, act=act)
        self.image_recon = conv(n_feats, args.n_colors, 3, padding=1, bias=True)
        self.edge_recon = conv(n_feats, 1, 3, padding=1, bias=True)
        self.add_mean1 = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.add_mean2 = common.MeanShift(1, rgb_mean, rgb_std, 1)

        self.image_head = nn.Sequential(*modules_head_image)
        self.edge_head = nn.Sequential(*modules_head_edge)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, image, edge):
        # image, edge = x
        image = self.sub_mean1(image)
        # edge = self.sub_mean2(edge)
        image = self.image_head(image)
        edge = self.edge_head(edge)
        # batch_size1, c1, h1, w1 = edge.shape
        # print(c1, h1, w1)
        # edge = self.tensor(edge)
        edge_ten = self.rb(edge)
        down_ten = image
        up_concat = []
        down_concat = []
        for i in range(self.n_groups):
            # print(i)
            up_ten = self.ups[i](down_ten)
            up_concat.append(up_ten)
            down_ten = self.downs[i](up_ten)
            down_concat.append(down_ten)
            down_cat = torch.cat(down_concat, 1)
            edge_ten = self.egems[i](edge_ten, down_cat)
            # edge_ten = self.egems[i](edge_ten, down_ten)
        up_ten = self.image_up(down_ten)
        up_concat.append(up_ten)
        img_up = torch.cat(up_concat, 1)
        img_fus = self.fusion(img_up)
        edge_up = self.edge_up(edge_ten)
        img_fus = self.jbf(img_fus, edge_up)
        img_sr = self.image_recon(img_fus)
        edge_sr = self.edge_recon(edge_up)
        img_sr = self.add_mean1(img_sr)
        # edge_sr = self.add_mean2(edge_sr)

        return img_sr, edge_sr

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
