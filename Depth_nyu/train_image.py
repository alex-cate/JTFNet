import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import imageio
import h5py
import os
from PIL import Image


def modcrop(img, scale):
    if len(img.shape) == 3:
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


# test_minmax = np.load('E:/SR/Code/2020IJCV/dkn-master/test_minmax.npy')
# f = h5py.File("E:/SR/dataset/depth/NYU/nyu_depth_v2_labeled.mat", 'r')
f = h5py.File("/data/xyp/TrainCode_bi/Data/DEPTH/nyu/nyu_depth_v2_labeled.mat", 'r')
depths = f["depths"]
depths = np.array(depths)
path_converted = 'D:/nyu/depth_hr/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

depths = depths.transpose((0, 2, 1))
sum_rmse = 0
rmse_list = []
for i in range(1000, len(depths)):
    # minmax = test_minmax[:, i-1000]
    # print(str(i + 1) + '.png')
    depths_img = depths[i]
    max = depths_img.max()
    min = depths_img.min()
    # print(max, min)
    depths_img = (depths_img - min) / (max - min) * 255
    sr = imageio.imread('/data/wh/2022TIP/result/Ours/nyu/depth_x16/' + str(i + 1) + '_x16.png')
    sr = modcrop(sr, 64)
    depths_img = modcrop(depths_img, 64)
    sr = (sr / 255) * (max - min) + min
    depths_img = (depths_img / 255) * (max - min) + min
    # sr = (sr / 255) * (min - max) + min
    rmse = np.sqrt(np.mean(np.power(depths_img - sr, 2)))
    rmse_list.append(rmse)
    sum_rmse = sum_rmse + rmse
avg_rmse = sum_rmse / (len(depths) - 1000)
variance = sum([(x - avg_rmse) ** 2 for x in rmse_list]) / (len(depths) - 1000) * 100
print('RMSE =', avg_rmse)
print('RMSE =', sum(rmse_list) / (len(depths) - 1000) * 100)
print('var =', variance)



# from PIL import Image
# import imageio
# import numpy as np

# depth_train_dir = '/DISK/wh/Data/DEPTH/nyu_data/test_depth.npy'
# image_train_dir = '/DISK/wh/Data/DEPTH/nyu_data/test_images_v2.npy'
# depth = np.load(depth_train_dir)
# image = np.load(image_train_dir)
# print(depth.shape)
# print(image.shape)
# for i in range(depth.shape[0]):
#     hr = depth[i]
#     color = image[i]
#     imageio.imsave('/DISK/wh/Data/DEPTH/nyu_data/testdata/depth_hr/{}.png'.format(i+1001), hr)
#     imageio.imsave('/DISK/wh/Data/DEPTH/nyu_data/testdata/color/{}.png'.format(i + 1001), color)
