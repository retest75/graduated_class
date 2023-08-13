# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:02:27 2023

@author: Chen ZE
"""

# Chap05.1.3-(1) Implement Dataset and DataLoader (1)
# folder: E:/graduate computer/PyTorch/Chap.05/image_and_image
# situation: when sample is image and label is image ,filename is in sequentially
# 樣本與標記都是圖片且檔名有順序
# all customized dataset must inherit Dataset
# include three basic method: __init__, __getitem__, __len__
# __init__: initialize
# __getiten__(): base on index to obtain data(根據index提取資料)
# __len__(): total number
# also can include other function to process image, like data augmentation



import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

#%% dataaset
class CustomizedDataset(Dataset):
    def __init__(self, root, data_transforms=None, target_transforms=None):
        """
        1.  root: folder and file load
        2.  transforms: image process
        """
        self.img = os.path.join(root, "image")  # image folder
        self.lbl = os.path.join(root, "label")  # lable folder
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        
        self.len_img = len(os.listdir(self.img)) # number of image
        self.len_lbl = len(os.listdir(self.lbl)) # number of label
        
        self.img = [f"{self.img}/{img}.png" for img in range(self.len_img)]
        self.lbl = [f"{self.lbl}/{lbl}.png" for lbl in range(self.len_lbl)]
    
    def __getitem__(self, index):
        img_path = self.img[index] # base in index to read image
        lbl_path = self.lbl[index] # base on index to read label
        
        image = Image.open(img_path)
        label = Image.open(lbl_path)
        
        if self.data_transforms:
            image = self.data_transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        return image, label
    
    def __len__(self):
        return len(self.img)

#%% dataloader
# dataloader will read data with batch
# image will be process when load data
# DataLoader(dataset, batch_size, shuffle)
# DataLoader will return a iterable object
        
batch_size = 5 # usually multiple of 4
root = "E:/graduate computer/PyTorch/Chap.05/image_and_image"
data_transforms = transforms.Compose([transforms.Resize(128),
                                      transforms.ToTensor()])
target_transforms = transforms.ToTensor()

train_dataset = CustomizedDataset(root, data_transforms=data_transforms, target_transforms=target_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

print(f"length of dataset: {len(train_dataset)}")
print('-'*15)

for img, lbl in train_dataloader:
    print(f"data size:  {img.size()}")   # [batch, channel, hight, width]=[5, 3, 128, 128]
    print(f"label size: {lbl.size()}\n") # [batch, channel, hight, width]=[5, 3, 256, 256]
    

