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
        # if args.ext.find('img')<0:
        #     path_bin=os.path.join(self.apath,'bin')
        #     os.makedirs(path_bin,exist_ok=True)

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr, self.images_color = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr, self.images_color = self._scan()
            # os.makedirs(self.dir_hr.replace(self.apath,path_bin),exist_ok=True)
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = imageio.imread(v)
                    # name_sep = v.replace(self.images_hr, os.path.join(self.images_hr, 'bin'))
                    # os.makedirs(name_sep, exist_ok=True)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for v in self.images_lr:
                    lr = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, lr)
                for v in self.images_color:
                    img = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                v.replace(self.ext, '.npy') for v in self.images_lr
            ]
            self.images_color = [
                v.replace(self.ext, '.npy') for v in self.images_color
            ]
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def __getitem__(self, idx):
        # lr, hr, filename = self._load_file(idx)
        # lr, hr, lr_e, hr_e = self._get_patch(lr, hr)
        # lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        # lr_tensor, hr_tensor, lr_tensor_e, hr_tensor_e = common.np2Tensor([lr, hr, lr_e, hr_e], self.args.rgb_range)
        # return lr_tensor, hr_tensor, lr_tensor_e, hr_tensor_e, filename

        depth_hr, depth_lr, color, filename = self._load_file(idx)
        depth_hr, depth_lr, color = self._get_patch(depth_hr, depth_lr, color)
        # depth_hr, depth_lr = common.set_channel([depth_hr, depth_lr], self.args.n_colors)
        # color = common.set_channel([color], 3)
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
            color = 0.2989 * color[:, :, 0] + 0.5870 * color[:, :, 1] + 0.1140 * color[:, :, 2]
            color = np.array(color)
            color = np.expand_dims(color, axis=2)
            depth_hr = np.expand_dims(depth_hr, axis=2)
            depth_lr = np.expand_dims(depth_lr, axis=2)
            depth_hr = depth_hr / 255.0
            depth_lr = depth_lr / 255.0
            color = color / 255.0
            # depth = skimage.io.imread(depth)
            # color = skimage.io.imread(color)
        elif self.args.ext.find('sep') >= 0:
            filename = depth_hr
            # print(depth_hr)
            # print(depth_lr)
            depth_hr = np.load(depth_hr)
            depth_lr = np.load(depth_lr)
            color = np.load(color)
            color = 0.2989 * color[:, :, 0] + 0.5870 * color[:, :, 1] + 0.1140 * color[:, :, 2]
            color = np.array(color)
            color = np.expand_dims(color, axis=2)
            depth_hr = np.expand_dims(depth_hr, axis=2)
            depth_lr = np.expand_dims(depth_lr, axis=2)
            depth_hr = depth_hr / 255.0
            depth_lr = depth_lr / 255.0
            color = color / 255.0
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return depth_hr, depth_lr, color, filename

    # def _process_img(self, depth_hr, depth_lr, color):
    #     if not self.train:
    #         # depth = common.modcrop(depth, self.scale)
    #         # color = common.modcrop(color, self.scale)
    #         hr_shape = depth_hr.shape
    #         depth_hr = depth_hr.reshape([int(hr_shape[0]), int(hr_shape[1]), 1])
    #         lr_shape = depth_lr.shape
    #         depth_lr = depth_lr.reshape([int(lr_shape[0]), int(lr_shape[1]), 1])
    #         # depth = np.array(depth)
    #         # color = np.array(color)
    #         depth_hr = depth_hr / 255.0
    #         depth_lr = depth_lr / 255.0
    #         color = color / 255.0
    #         # depth_lr = common.gen_lr(depth, self.scale,True)
    #         # imageio.imsave('/DISK/wh/Depth/test_img/{}_hr.png'.format(self.name), depth*255)
    #         # imageio.imsave('/DISK/wh/Depth/test_img/cones_lr_return.png', depth_lr * 255)
    #     else:
    #         depth_hr, depth_lr, color = common.augment([depth_hr, depth_lr, color])
    #
    #     return depth_hr, depth_lr, color

    def _get_patch(self, depth_hr, depth_lr, color):
        patch_size = self.args.patch_size
        scale = self.scale
        if self.train:
            depth_lr, depth_hr, color = common.get_patch(
                depth_lr, depth_hr, color, patch_size, scale
            )
            depth_lr, depth_hr, color = common.augment([depth_lr, depth_hr, color])
            # lr = common.add_noise(lr, self.args.noise)
        else:
            # depth_hr = common.modcrop(depth_hr, 64)
            # depth_lr = common.modcrop(depth_lr, 4)
            # color = common.modcrop(color, 64)
            ih, iw = depth_lr.shape[0:2]
            depth_hr = depth_hr[0:ih * scale, 0:iw * scale]

        return depth_hr, depth_lr, color

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
