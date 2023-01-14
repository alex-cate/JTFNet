import os
from data import common
from data import srdata
import numpy as np
import torch
import torch.utils.data as data
class Testdata(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Testdata, self).__init__(args, name=name, train=train, benchmark=True)
    def _scan(self):
        list_hr = []
        list_lr = []
        list_color = []
        for entry in os.scandir(self.dir_hr):
            # print(entry.name)
            filename = os.path.splitext(entry.name)[0]
            print(filename)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            list_lr.append(os.path.join(self.dir_lr, filename + self.ext))
            list_color.append(os.path.join(self.dir_color, filename + self.ext))
            # print(list_hr)
        list_hr.sort()
        list_lr.sort()
        list_color.sort()
        return list_hr, list_lr, list_color

    def _set_filesystem(self, dir_data):
        # self.apath = os.path.join(dir_data, 'tof')
        self.apath = os.path.join(dir_data, 'testdata')
        # self.apath = os.path.join(dir_data, 'testdata', 'test')
        self.dir_hr = os.path.join(self.apath, 'depth_hr')
        self.dir_lr = os.path.join(self.apath, 'depth_x{}'.format(self.scale))
        self.dir_color = os.path.join(self.apath, 'color')
        self.ext = '.png'







