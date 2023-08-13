# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:37:43 2023

@author: Chen ZE
"""

# Chap05.1.3-(2) Implement Dataset and DataLoader (2)
# folder: E:/graduate computer/PyTorch/Chap.05/image_csv
# label was save to csv
# we can base on filename in csv to read image

#import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


root = "E:/graduate computer/PyTorch/Chap.05/image_csv"
df = pd.read_csv(os.path.join(root, "train.csv"))
x = df["ID"].values
y = df["Label"].values

print(df.head())
#print(x[:3])

# another method to convert series into numpy
#x = df["ID"].to_numpy()
#y = df["Label"].to_numpy()

#%%
class CustomizedDataset(Dataset):
    def __init__(self, root, x, y, transforms=None):
        self.root = os.path.join(root, "train_images")
        self.img = [f"{os.path.join(self.root, path)}" for path in x]
        self.lbl = [i for i in y]
        self.transforms = transforms
    
    def __getitem__(self, index):
        image_path = self.img[index]
        label = self.lbl[index]
        image = Image.open(image_path)
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, label
    
    def __len__(self):
        return len(self.img)

#%%
img_list = []
lbl_list = []
batch_size = 512
transforms = transforms.Compose([transforms.CenterCrop(256),
                                 transforms.ToTensor()])

dataset = CustomizedDataset(root, x, y, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

print(f"number of dataset: {len(dataset)}")
for index, (image, label) in enumerate(dataloader):
    print(f"times: {index+1}/5")
    img_list.append(image)
    lbl_list.append(label)
    
a = img_list[0] # shape = [512, 1, 256, 256] first batch data
print(a.shape)
b = lbl_list[0] # shape = [512] fitsr batch label
print(b.shape)

# chack
second_root = "E:/graduate computer/PyTorch/Chap.05/image_csv/train_images/train_00001.png" # second image route
image = Image.open(second_root)
second = transforms(image)

print(a[1, :]==second) # True, show second image
print(b[1])




        