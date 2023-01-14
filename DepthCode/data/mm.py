import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class MM(srdata.SRData):
    def __init__(self, args,name='MM', train=True):
        super(MM, self).__init__(args,name=name, train=train)
        # self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.repeat = 1

    def _scan(self):
        list_hr = []
        list_lr = []
        list_color = []
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename_hr = 'image_hr_{:0>6}'.format(i)
            filename_lr = 'image_x{}_{:0>6}'.format(self.scale,i)
            filename_color = 'image_color_{:0>6}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename_hr + self.ext))
            list_lr.append(os.path.join(self.dir_lr, filename_lr + self.ext))
            list_color.append(os.path.join(self.dir_color, filename_color + self.ext))
            list_hr.sort()
            list_lr.sort()
            list_color.sort()
        return list_hr, list_lr, list_color

    def _set_filesystem(self, dir_data):
        # self.apath = os.path.join(dir_data, 'traindata_p16')
        self.apath = os.path.join(dir_data, 'traindata')
        self.dir_hr = os.path.join(self.apath, 'depth_hr')
        self.dir_lr = os.path.join(self.apath, 'depth_x{}'.format(self.scale))
        self.dir_color = os.path.join(self.apath, 'color')
        self.ext = '.npy'

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx
