# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:09:53 2023

@author: Chen ZE
"""

# Chap05.1.2 Transform
# PyTorch include a lot of transform function
# input must be PIL
# reference: https://pytorch.org/vision/stable/transforms.html#

import torch
from torchvision import transforms
from PIL import Image

#path = "E:/graduate computer/PyTorch/prace_image/practice_1.PNG"  # yolo
path = "E:/graduate computer/PyTorch/prace_image/practice_3.jpg"  # weiwei


#%% read image
img = Image.open(path)
print(f"size = {img.size}") # (W, H)
#img.show()

#%% 放大/縮小
resize = transforms.Resize(512) # if parameter is int, then smaller side will be adjust
img_resize = resize(img)
print(f"size = {img_resize.size}")
#img_resize.show()

#%% 自中心剪裁
crop = transforms.CenterCrop(512)
img_crop = crop(img)
print(f"size = {img_crop.size}")
#img_crop.show()

#%% 自左上/右上/左下/右下/中心為參考點剪裁
five_crop = transforms.FiveCrop(512)
img_five_crop = five_crop(img) # return 5 imagw by tuple
print(f"size = {img_five_crop[0].size}")
#img_five_crop[4].show()

#%% 灰階
grayscale = transforms.Grayscale(1)
img_gray = grayscale(img)
print(f"size = {img_gray.size}")
#print(f"mode = {img_gray.mode}") if Grayscale() then L, Grayscale(3) then RGB
#img_gray.show()

#%% 轉tensor
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)
#img_tensor = img_tensor.long()       # float32 -> floar64
print(f"size = {img_tensor.size()}")  # (C, H, W) in PyTorch
print(f"data type = {img_tensor.dtype}")
#print(img_tensor)

#%% compose
transforms = transforms.Compose([transforms.Resize(512),
                                 transforms.CenterCrop(256),
                                 transforms.Grayscale(),
                                 transforms.ToTensor()
                                ])
img_compose = transforms(img)
#img_compose.show()
print(img_compose.size()) # (C, H, W) = (1, 256, 256)

