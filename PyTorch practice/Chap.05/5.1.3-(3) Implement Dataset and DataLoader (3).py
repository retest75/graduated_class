# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:14:20 2023

@author: Chen ZE
"""
""" 報錯 """
# Chap05.1.3-(3) Implement Dataset and DataLoader (3)
# path: "E:/graduate computer/PyTorch/Chap.05/image_folder"
# use ImageFolder
# all same class be save in one folder which name is class, and filenme is in sequential
# ImageFolder include three parameter
# (1) root: path
# (2) transform: transforms for sample image
# (3) target_transform: transforms for class image
# ImageFolder include three arrtibutes
# (1) self.classes: list, used to save fold(class) name
# (2) self.class_to_idx: dictionary, appear mapping from class to label
# (3) self.imgs: list, show all image and its label by tuple [(image_path, label)]

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

root = "E:/graduate computer/PyTorch/Chap.05/image_folder"
#%%
transforms = transforms.Compose([transforms.CenterCrop(256),
                                 transforms.Grayscale(),
                                 transforms.ToTensor()])
# if we don't convert into tensor, it will return PIL format
dataset = ImageFolder(root, transform=transforms, target_transform=transforms)
#print(dataset.classes)       # ["NORMAL", "PNEUMONIA"]
#print(dataset.class_to_idx)  # {"NORMAL":0, "PNEUMONIA":1}
#print(dataset.imgs[:3])      # (image_path, label)
#print(len(dataset.imgs))     # 5216

#%%
batch_size = 512
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
for data, label in dataloader:
   print(data)