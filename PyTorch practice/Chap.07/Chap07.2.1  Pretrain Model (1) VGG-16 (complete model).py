# -*- coding: utf-8 -*-
"""
Created on Sat May  6 21:06:13 2023

@author: Chen ZE
"""

# Chap07.2.1  Pretrain Model (1) VGG-16
# use complete model
# old version is ptrtrained=True(deprecated)
# new version must import corresponding weight, such like ResNet_Weight
# reference: https://pytorch.org/vision/stable/models.html
# increase one dimension can use unsqueeze
# reference: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch-unsqueeze
# index of ImageNet can see here: https://blog.csdn.net/u013491950/article/details/83927968

import torch
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights # import corresponding weight
from torchvision import transforms
from torch.nn import functional as F
from torchsummary import summary

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.__version__)

#%% load pretrained model and its weight
#model = vgg16(weights=None)                 # didn't load trained weights
model = vgg16(weights=VGG16_Weights.DEFAULT)
model = model.to(device)                     # convert to GPU
summary(model, input_size=(3, 224, 224))     # show model frame

#%% read image
path = "E:/graduate computer/PyTorch/cock.jpg" # cock(公雞)
img = Image.open(path)
#img.show()

#%% preprocess image
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                ])
img_tensor = transforms(img)
# resize to [batch, C, H, W] = [1, 3, 224, 224]
img_batch = img_tensor.unsqueeze(0).to(device)
#print(img_batch.size()) # [1, 3, 224, 224]

#%% input img to test
model.eval()
with torch.no_grad():
    output = model(img_batch) # output size: [batch, num_classes] = [1, 1000]

y_pred = F.softmax(output[0], dim=0)
#print(y_pred.size()) # size = [1000]
print(y_pred) # show porbability of each corresponding index

pred = torch.argmax(y_pred)
prob = torch.max(y_pred)
print(f"prediction index: {pred}") # if index = 7 , that is cock(公雞)
print(f"predicted probability: {prob}")


