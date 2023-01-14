import os

from data import common

import numpy as np
import imageio
import skimage

import torch
import torch.utils.data as data


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.name = name
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale[0]
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr, self.images_color = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr, self.images_color = self._scan()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def __getitem__(self, idx):
        depth_hr, depth_lr, color, filename = self._load_file(idx)
        depth_hr, depth_lr, color = self._process_img(depth_hr, depth_lr, color)
        depth_hr, depth_lr, color = common.np2Tensor([depth_hr, depth_lr, color], self.args.rgb_range)
        # print(filename)
        return depth_hr, depth_lr, color, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        color = self.images_color[idx]
        depth_hr = self.images_hr[idx]
        depth_lr = self.images_lr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = depth_hr
            depth_hr = imageio.imread(depth_hr)
            depth_lr = imageio.imread(depth_lr)
            color = imageio.imread(color)
        elif self.args.ext.find('sep') >= 0:
            filename = depth_hr
            depth_hr = np.load(depth_hr)
            depth_lr = np.load(depth_lr)
            color = np.load(color)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return depth_hr, depth_lr, color, filename

    def _process_img(self, depth_hr, depth_lr, color):
        if not self.train:
            # depth_hr = common.modcrop(depth_hr, 64)
            # depth_lr = common.modcrop(depth_lr, 4)
            # color = common.modcrop(color, 64)
            hr_shape = depth_hr.shape
            depth_hr = depth_hr.reshape([int(hr_shape[0]), int(hr_shape[1]), 1])
            lr_shape = depth_lr.shape
            depth_lr = depth_lr.reshape([int(lr_shape[0]), int(lr_shape[1]), 1])
            depth_hr = depth_hr / 255.0
            depth_lr = depth_lr / 255.0
            color = color / 255.0
        else:
            depth_hr, depth_lr, color = common.augment([depth_hr, depth_lr, color])

        return depth_hr, depth_lr, color

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
