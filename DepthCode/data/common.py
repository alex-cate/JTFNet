import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
from skimage import filters
import skimage.transform as st
import cv2
import torch
import imageio
from torchvision import transforms

def gen_lr(depth, scale,test=False):
    hr_shape = depth.shape
    lr = cv2.resize(depth, (int(hr_shape[1] / scale), int(hr_shape[0] / scale)), interpolation=cv2.INTER_CUBIC)
    lr = lr.reshape([int(hr_shape[0] / scale), int(hr_shape[1] / scale), 1])
    # if test:
    #     imageio.imsave('/DISK/wh/Depth/test_img/cones_lr.png', lr * 255)
    return lr

def normalize(img, c):
    data = [(img[:, :, i] - min(img[:, :, i])) / (max(img[:, :, i]) - min(img[:, :, i])) for i in range(c)]
    return data

def modcrop(img, scale):
    if len(img.shape) == 3:  # 高度、宽度、通道数
        h, w, _ = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w]
    return img

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # print(img.shape)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        # tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def add_noise(x, noise='.'):
    if noise!='.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]
