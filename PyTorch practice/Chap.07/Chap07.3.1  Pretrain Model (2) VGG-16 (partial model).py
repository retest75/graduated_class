# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:02:12 2023

@author: Chen ZE
"""

# Chap07.3.1  Pretrain Model (2) VGG-16
# use partial model
# reference: https://www.jianshu.com/p/a4c745b6ea9b

import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from torch import nn
from torchsummary import summary
from PIL import Image

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% load pretrain model and observe its frame
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

for layer in model.modules(): # self.modules() is  generator ,will visit "every submodule"
    print(layer)              # the VGG-16 model include three submodule
                              # first: features. second: average pooling, third: classifier
                              # we only need first and second submodule (只作feature extration)

#%% modify pretrain model
class ModifiedModel(nn.Module):
    def __init__(self, pretrained_model, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained_model = pretrained_model
        self.children_list = []
        
        # visit first submodule
        # self.children() also is generator , will visit every "first submodule"
        # self.name_children() will named each submodule
        for name, layer in self.pretrained_model.named_children():
            self.children_list.append(layer)
            if name == self.output_layer: # until find some particular submodule, then break
                print("Found !!")
                break
        
        # construct new model
        self.net = nn.Sequential(*self.children_list) # * used to unpack a iterable object into a sequential
        self.pretrained_model = None                  # release space
    
    def forward(self, x):
        x = self.net(x)
        return x

model = ModifiedModel(model, "avgpool") # only want features and avgpool and exclude classfier
model = model.to(device)

#%% observe modified model
for layer in model.named_children():
    print(layer)

#%% read any image
path = "E:/graduate computer/PyTorch/cock.jpg" # cock(公雞)
img = Image.open(path)
#img.show()

transforms = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                ])
img_to_tensor = transforms(img)
img_input = img_to_tensor.unsqueeze(0).to(device) # size = [batch, C, H, W]
#print(img_input.size()) # [1, 3, 224, 224]

#%% test
model.eval()
with torch.no_grad():
    output = model(img_input)

print(output)
