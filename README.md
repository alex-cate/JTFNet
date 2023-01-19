# JTFNet
# JTFNet: Joint Trilateral Filter Learning for Guided Depth Map Super-resolution
# Abstract
Most state of the arts (SOTAs) of depth map super-resolution cannot adaptively tune the guidance fusion for all feature positions by channel-wise feature concatenation with spatially sharing convolutional kernels. This paper proposes JTFNet to resolve this issue, which simulates the traditional Joint Trilateral Filter (JTF). Specifically, a novel JTF block is introduced to adaptively tune the fusion pattern between color features and depth features for all feature positions. Moreover, based on the variant of JTF block whose target features and guidance features are in the cross-scale shape, the fusion for depth features is performed in a bi-directional way. Therefore, the error accumulation along scales can be effectively mitigated by iteratively HR feature guidance. Compared with SOTAs, the sufficient experiment is conducted on mainstream synthetic datasets and real datasets, i.e. Middlebury, NYU and ToF-Mark, which shows remarkable improvement of our JTFNet.

# Dependencies
Python 3.8  
PyTorch >= 1.0.0  
numpy  
skimage  
imageio  
matplotlib  
tqdm  
tensorboardX  
cv2 >= 3.xx   

# Trian
```Bash 
python main.py --model .. --save .. –scale ..[2,4,8,16] --save_models --save_results --batch_size .. --data_test ..
```

# Resume
```Bash 
python main.py --model .. --save .. –scale .. --save_models --save_results --batch_size .. --load .. --resume ..
```

# Test
```Bash
python main.py --model .. --save .. --scale .. --save_results --test_only --pre_train .. --date_test ..
```
