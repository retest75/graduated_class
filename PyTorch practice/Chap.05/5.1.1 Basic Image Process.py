# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:09:53 2023

@author: Chen ZE
"""

# Chap05.1.1 Basic Image Process


import numpy as np
from PIL import Image
import cv2

#path = "E:/graduate computer/PyTorch/prace_image/practice_1.PNG"  # yolo0
#path = "E:/graduate computer/PyTorch/prace_image/practice_2.jpg"  # yolo1
#path = "E:/graduate computer/PyTorch/prace_image/practice_3.jpg"  # weiwei
path = "E:/graduate computer/PyTorch/prace_image/practice_4.jpg"  # Marin
#path = "E:/graduate computer/PyTorch/prace_image/practice_5.jpg"  # Chizuru
#path = "E:/graduate computer/PyTorch/prace_image/practice_6.JPG"  # cat1
#path = "E:/graduate computer/PyTorch/prace_image/practice_7.jpg"  # cat2

#%% image process (1): PIL
img_PIL = Image.open(path) # read image
print(f"data type: {type(img_PIL)}")  # <class 'PIL.PngImagePlugin.PngImageFile'>
print(f"format: {img_PIL.format}")    # 副檔名
print(f"size: {img_PIL.size}")        # (width, hight)
print(f"mode: {img_PIL.mode}")        # RGB
#img_PIL.show()                        # show image

#%% image process (2): open cv
# when read image by open cv, it will be save by ndarray
# channel is ipen cv is "BGR", NOT "RGB"
img_cv = cv2.imread(path)                                  # default: BGR
img_cv_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)      # grayscle 
print(f"data type: {type(img_cv)}")         # <class 'numpy.ndarray'>
print(f"data type: {type(img_cv_gray)}")
print(f"shape: {img_cv.shape}")             # (hight, width, channel)
print(f"shape: {img_cv_gray.shape}")

#cv2.namedWindow('marin', cv2.WINDOW_NORMAL) # adjust size of window (條整顯示視窗大小)
cv2.imshow("marin", img_cv)                 # show image with some file name
cv2.waitKey(0)                              # open image untile enter any key
cv2.destroyWindow("marin")                  # when accept above key then close file

