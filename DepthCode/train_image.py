# import numpy as np
# import os
# import glob
# import imageio
# import random
#
#
# def modcrop(img, scale):
#     if len(img.shape) == 3:  # 高度、宽度、通道数
#         h, w, _ = img.shape
#         h = int((h / scale)) * scale
#         w = int((w / scale)) * scale
#         img = img[0:h, 0:w, :]
#     else:
#         h, w = img.shape
#         h = int((h / scale)) * scale
#         w = int((w / scale)) * scale
#         img = img[0:h, 0:w]
#     return img
#
#
# def augment(img, type):
#     if type < 0.25:
#         img = img[:, ::-1, :]
#     elif type >= 0.75:
#         img = img[::-1, :, :]
#     elif type >= 0.25 and type < 0.5:
#         img = img.transpose(1, 0, 2)
#     else:
#         img = img[::-1, :, :]
#         img = img[:, ::-1, :]
#
#     return img
#
#
# def normalize(img, c):
#     data = [(img[:, :, i] - min(img[:, :, i])) / (max(img[:, :, i]) - min(img[:, :, i])) for i in range(c)]
#     return data
#
#
# def get_patch(depth_img, x2_img, x4_img, x8_img, x16_img, color_img, h, w, patch_size, stride, sum):
#     times = 0
#     for x in range(0, h - patch_size + 1, stride):
#         for y in range(0, w - patch_size + 1, stride):
#             # 滑动窗口取样
#             sub_depth_img = depth_img[x*16: (x + patch_size)*16, y*16: (y + patch_size)*16]
#             sub_x2_img = x2_img[x*8: (x + patch_size)*8, y*8: (y + patch_size)*8]
#             sub_x4_img = x4_img[x*4: (x + patch_size)*4, y*4: (y + patch_size)*4]
#             sub_x8_img = x8_img[x*2: (x + patch_size)*2, y*2: (y + patch_size)*2]
#             sub_x16_img = x16_img[x:x + patch_size, y: y + patch_size]
#             sub_color_img = color_img[x*16: (x + patch_size)*16, y*16: (y + patch_size)*16]
#
#             if (np.max(sub_depth_img) - np.min(sub_depth_img)) != 0 and (
#                     np.max(sub_color_img) - np.min(sub_color_img)) != 0:
#                 sub_depth_img = sub_depth_img.reshape([patch_size*16, patch_size*16, 1])
#                 sub_x2_img = sub_x2_img.reshape([patch_size*8, patch_size*8, 1])
#                 sub_x4_img = sub_x4_img.reshape([patch_size*4, patch_size*4, 1])
#                 sub_x8_img = sub_x8_img.reshape([patch_size*2, patch_size*2, 1])
#                 sub_x16_img = sub_x16_img.reshape([patch_size, patch_size, 1])
#                 sub_color_img = sub_color_img.reshape([patch_size*16, patch_size*16, 3])
#
#                 sub_depth_img = sub_depth_img / 255.0
#                 sub_x2_img = sub_x2_img / 255.0
#                 sub_x4_img = sub_x4_img / 255.0
#                 sub_x8_img = sub_x8_img / 255.0
#                 sub_x16_img = sub_x16_img / 255.0
#                 sub_color_img = sub_color_img / 255.0
#                 # sub_depth_img = normalize(sub_depth_img, 1)
#                 # sub_color_img = normalize(sub_color_img, 3)
#
#                 times += 1
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_hr",
#                                  "image_hr_" + "{:0>6}".format(sum + times) + ".npy"),
#                     sub_depth_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x2",
#                                  "image_x2_" + "{:0>6}".format(sum + times) + ".npy"),
#                     sub_x2_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x4",
#                                  "image_x4_" + "{:0>6}".format(sum + times) + ".npy"),
#                     sub_x4_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x8",
#                                  "image_x8_" + "{:0>6}".format(sum + times) + ".npy"),
#                     sub_x8_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x16",
#                                  "image_x16_" + "{:0>6}".format(sum + times) + ".npy"),
#                     sub_x16_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/color",
#                                  "image_color_" + "{:0>6}".format(sum + times) + ".npy"),
#                     sub_color_img)
#
#                 type = random.uniform(0, 1)
#                 aug_depth_img = augment(sub_depth_img, type)
#                 aug_x2_img = augment(sub_x2_img, type)
#                 aug_x4_img = augment(sub_x4_img, type)
#                 aug_x8_img = augment(sub_x8_img, type)
#                 aug_x16_img = augment(sub_x16_img, type)
#                 aug_color_img = augment(sub_color_img, type)
#                 times += 1
#
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_hr",
#                                  "image_hr_" + "{:0>6}".format(sum + times) + ".npy"),
#                     aug_depth_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x2",
#                                  "image_x2_" + "{:0>6}".format(sum + times) + ".npy"),
#                     aug_x2_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x4",
#                                  "image_x4_" + "{:0>6}".format(sum + times) + ".npy"),
#                     aug_x4_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x8",
#                                  "image_x8_" + "{:0>6}".format(sum + times) + ".npy"),
#                     aug_x8_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/depth_x16",
#                                  "image_x16_" + "{:0>6}".format(sum + times) + ".npy"),
#                     aug_x16_img)
#                 np.save(
#                     os.path.join("/DISK/wh/Data/DEPTH/noise/traindata/color",
#                                  "image_color_" + "{:0>6}".format(sum + times) + ".npy"),
#                     aug_color_img)
#     return times
#
#
# patch_size = 8
# stride = 4
# # hr_train_dir = '/media/h428_zuo/本地磁盘1/zuo/gdsr/depth_patches.h5'
# hr_train_dir = '/DISK/wh/Data/DEPTH/noise_data/traindata/depth_hr'
# color_train_dir = '/DISK/wh/Data/DEPTH/noise_data/traindata/color'
# x2_train_dir = '/DISK/wh/Data/DEPTH/noise_data/traindata/depth_x2'
# x4_train_dir = '/DISK/wh/Data/DEPTH/noise_data/traindata/depth_x4'
# x8_train_dir = '/DISK/wh/Data/DEPTH/noise_data/traindata/depth_x8'
# x16_train_dir = '/DISK/wh/Data/DEPTH/noise_data/traindata/depth_x16'
#
# hr_list = sorted(glob.glob(os.path.join(hr_train_dir, "*.png")))
# x2_list = sorted(glob.glob(os.path.join(x2_train_dir, "*.png")))
# x4_list = sorted(glob.glob(os.path.join(x4_train_dir, "*.png")))
# x8_list = sorted(glob.glob(os.path.join(x8_train_dir, "*.png")))
# x16_list = sorted(glob.glob(os.path.join(x16_train_dir, "*.png")))
# color_list = sorted(glob.glob(os.path.join(color_train_dir, "*.png")))
#
# sum = 0
# for i in range(len(hr_list)):
#     # print(i)
#     hr_img = imageio.imread(hr_list[i])
#     x2_img = imageio.imread(x2_list[i])
#     x4_img = imageio.imread(x4_list[i])
#     x8_img = imageio.imread(x8_list[i])
#     x16_img = imageio.imread(x16_list[i])
#     color_img = imageio.imread(color_list[i])
#
#     # depth_img = modcrop(depth_img, stride)
#     # color_img = modcrop(color_img, stride)
#     print(hr_img.shape)
#     print(color_img.shape)
#
#     if len(x16_img.shape) == 3:
#         h, w, c = x16_img.shape
#     else:
#         h, w = x16_img.shape
#
#     times = get_patch(hr_img, x2_img, x4_img, x8_img, x16_img, color_img, h, w, patch_size, stride, sum)
#     sum += times
#     print("the patch number of image %d is %d, sum of patch is %d " % (i + 1, times, sum))

import imageio
import numpy as np
import glob
import os

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

sr_train_dir = '/DISK/wh/experiment_depth/depth_x4_old/results/'
# sr_train_dir = '/DISK/wh/experiment_depth/depth_x16_old_p16/results/'
hr_train_dir = '/DISK/wh/Data/DEPTH/testdata/depth_hr/'
sr_list = sorted(glob.glob(os.path.join(sr_train_dir, "*.png")))
hr_list = sorted(glob.glob(os.path.join(hr_train_dir, "*.png")))
scale = 32
for i in range(len(sr_list)):
    sr = imageio.imread(sr_list[i])
    hr = imageio.imread(hr_list[i])

    # hr = modcrop(hr, scale)
    # sr = modcrop(sr, scale)
    # valid=sr-hr
    # rmse=np.pow(valid,2).mean().sqrt()
    # rmse=np.sqrt(np.mean((sr - hr) ** 2))
    # mad=np.mean(np.abs(sr - hr))

    sr = sr.astype(np.double)
    hr = hr.astype(np.double)
    max = hr.max()
    min=hr.min()

    diff = hr - sr

    h, w = diff.shape
    mad = np.sum(abs(diff)) / (h * w)
    rmse = np.sqrt(np.sum(np.power(diff, 2)/(h*w)))
    # print(mad)
    # print(sr_list[i].split('/')[-1],mad,rmse)
    print(sr_list[i].split('/')[-1], 'MAD = {:.4f} RMSE = {:.4f}'.format(mad, rmse))
    # print(rmse)


# import imageio
# import numpy as np
# import glob
# import os

# depth_hr = np.load("/DISK/wh/Data/nyu_data/test_depth.npy")
# # depth_lr = np.load(depth_lr)
# color = np.load("/DISK/wh/Data/nyu_data/test_images_v2.npy")
# print(color.shape)
# # color = np.load("/DISK/wh/Data/nyu_data/testdata/color/")
# for i in range(depth_hr.shape[0]):
#     depth=depth_hr[i]
#     img=color[i]
#
#     imageio.imsave('/DISK/wh/Data/nyu_data/testdata/color/{}.png'.format(i), img)
#     imageio.imsave('/DISK/wh/Data/nyu_data/testdata/depth_hr/{}.png'.format(i), depth)