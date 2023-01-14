import os
from data import common
from data import srdata
import numpy as np
import torch
import torch.utils.data as data


class Benchmark(srdata.SRData):
    # def __init__(self, args, train=True):
    #     super(Benchmark, self).__init__(args, train, benchmark=True)

    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(args, name=name, train=train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = []
        list_color = []
        list_hr.append(os.path.join(self.dir_hr, self.name + self.ext))
        list_lr.append(os.path.join(self.dir_lr, self.name + self.ext))
        list_color.append(os.path.join(self.dir_color, self.name + self.ext))
        # print(list_depth)
        # print(list_color)
        return list_hr, list_lr, list_color

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'testdata')
        self.dir_hr = os.path.join(self.apath, 'depth_hr')
        self.dir_lr = os.path.join(self.apath, 'depth_x{}'.format(self.scale))
        self.dir_color = os.path.join(self.apath, 'color')
        self.ext = '.png'
