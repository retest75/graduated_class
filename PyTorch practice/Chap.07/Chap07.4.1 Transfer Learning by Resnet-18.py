# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:56:19 2023

@author: Chen ZE
"""

# Chap07.4.1 Transfer Learning by Resnet-18
# reference (make_grid): https://www.bing.com/search?q=make_grid+pytorch&form=ANNTH1&refig=6e2a41960cd04fe7ac051e7e85611d3d&sp=1&lq=0&qs=HS&pq=make&sc=10-4&cvid=6e2a41960cd04fe7ac051e7e85611d3d

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


print(torch.__version__)
print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "E:/graduate computer/PyTorch/dataset/hymenoptera_data"
#train_path = os.path.join(root, "train")
#val_path = os.path.join(root, "val")

#%% Customed Dataset(use ImageFolder)
batch_size = 4

data_transforms = {
    "train":transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                ]),
    "val":transforms.Compose([transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                              ])
    }
image_datasets = {x:ImageFolder(os.path.join(root, x), data_transforms[x])
                  for x in ["train", "val"]}

dataloader = {x:DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
              for x in ["train", "val"]}

for x in ["train", "val"]:
    print(f"{x} data number: {len(image_datasets[x])}")
    print(f"{x} data classes: {image_datasets[x].classes}")

classes = image_datasets["train"].class_to_idx
#print(classes)     # {"ants":0, "bees":1}
class_names = image_datasets['train'].classes
#print(class_names) # ["ants", "bees"]

#%% Visualization Image
from torchvision.utils import make_grid # just show image

def imshow(img, title=None):
    img = img.numpy().transpose(1, 2, 0)   # [C, H, W] -> [H, W, C]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean                 # since dataloader have already normalize, here is back normalize
    img = np.clip(img, 0, 1)
    plt.axis("off")
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# read a batch image
image, target = next(iter(dataloader["train"]))
#print(image.size()) # [batch, C, H, W] = [4, 3, 224, 224]

# show a batch image
output = make_grid(image, padding=10, padding_value=1)
#print(output.size())# [C, H + padding*2(上下), W + padding*(nrow+1)(左右)] = [3, 244, 946]
imshow(output, title=[class_names[x] for x in target])

#%% Define Train Model
import time
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_weight = copy.deepcopy(model.state_dict()) # duplicate parameters of model
    best_acc = 0.0
    
    # start to train
    for epoch in range(1, num_epochs+1):
        print(f"Epoch: {epoch}/{num_epochs}")
        print("-" * 15)
        
        # each epoch have train and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # base on batch size to train or validate
            for img, target in dataloader[phase]:
                img, target = img.to(device), target.to(device)
                
                # zero the parameters gradient
                optimizer.zero_grad()
                
                # in train phase, model must operate gradient descent
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(img)
                    
                    loss = criterion(outputs, target)
                    _, pred = torch.max(outputs, 1) # predicted label
                    
                    if phase == "train":
                        loss.backward()  # backpropagation
                        optimizer.step() # update patameter
                    
                running_loss += loss.item() * img.size(0) # 看不懂這裡
                running_corrects += torch.sum(pred == target.data)
                
            # control scheduler(learning decrease)
            if phase == "train":
                scheduler.step()
            
            epoch_loss = running_loss / len(image_datasets[phase]) # calculate average loss
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weight = copy.deepcopy(model.state_dict()) # save best parameters
            
        print()
    
    # calculate time
    times = time.time() - since
    print(f"Train Complete in {(times/60):.0f} min {(times%60):.0f} sec")
    print(f"Best_acc: {best_acc}")
    
    # load best parameter to current model
    model.load_state_dict(best_model_weight)
    
    return model

#%% Define Visualization Function
def show_result(img, title=None):
    img = img.numpy().transpose(1, 2, 0)  # [Batch, C, H, W] -> [Batch, H, W, C]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)

def visualize_model(model, num_image=6):
    # save current mode
    was_training = model.training # boolean, True: train mode, False: evaluation mode
    
    model.eval()
    image_so_far = 0
    fig = plt.figure()
    
    with torch.no_grad():
        for i, (img, target) in enumerate(dataloader["val"]):
            img, target = img.to(device), target.to(device)
            
            output = model(img)
            _, pred = torch.max(output, 1)
            
            # visualize prediction image and label
            for j in range(img.size()[0]): # batch size
                image_so_far += 1
                plt.subplot(num_image//4+1, 4, image_so_far)
                plt.axis("off")
                plt.title(class_names[pred[j]])
                show_result(img.cpu().data[j])
                
                if image_so_far == num_image:
                    model.train(mode=was_training) # back to original mode
                    return
                
        model.train(model=was_training)
    plt.tight_layout()
    plt.show()

#%% Define Pretrained Model
from torchvision.models import resnet18, ResNet18_Weights
from torch import optim
from torch import nn
from torch.optim import lr_scheduler

model = resnet18()
weights = ResNet18_Weights.DEFAULT
model_ft = resnet18(weights=weights)

# observe frame of ResNet18
#for layer in model_ft.modules():
#    print(layer)

# connect self-defing fully connected layer
in_features = model_ft.fc.in_features    # 512, that is input features in ResNet = 18
model_ft.fc = nn.Linear(in_features, 2)  # connect new fully connected layer

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# learning rate will decrease after each 7 epochs 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# observe new model frame
for layer in model_ft.modules():
    print(layer)

#%% Start Train
model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

# Visualize Result
visualize_model(model_ft)

#%% Set Pretrained Model Not Train(預先訓練模型不用再訓練)
model_conv = resnet18(weights=weights)

# parameters of pretrained model didn't participate gradient descent
for param in model_conv.parameters():
    param.requirs_grad=False

in_features = model_conv.fc.in_features
model_conv.fc = nn.Linear(in_features, 2)

model_conv = model_conv.to(device)

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
visualize_model(model_conv)



    
    
                    


