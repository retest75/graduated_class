# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:01:44 2023

@author: Chen ZE
"""

# Chap04.2.1 MNIST for nn.funtional
# nn.squential be applieded to simple model
# nn.functional be applied to complicated model
# function of nn.functional is lower case, nn.squential is upper case
# usually use class to define model in nn.functional
# compare with part of "start train" in 4.1.1
# reference: https://pytorch.org/docs/stable/nn.functional.html

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

#%% set parameter
path_train = "E:/graduate computer/PyTorch/dataset/MNIST/train" # save dataset
path_test = "E:/graduate computer/PyTorch/dataset/MNIST/test"
batch_size = 600
#print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if can use GPU

#%% feature scale
#transform = transforms.Compose([transforms.ToTensor(),
#                                transforms.Normalize(0.1307, 0.3081)])


#%% download dataset
# train/test dataset
train_set = MNIST(path_train, train=True, download=True, transform=transforms.ToTensor())
test_set = MNIST(path_test, train=False, download=True, transform=transforms.ToTensor())
#print(f"shape of train dataset: {train_set.data.shape}") # [60000, 28, 28], 60000筆資料
#print(f"shape of test dataaset: {test_set.data.shape}")  # [10000, 28, 28], 10000筆資料

#%% show data
# dataset is grayscale graph, so 0 is white, 255 is black
#print(train_set.targets[:10])          # first 10 ground truth
first_data = train_set.data[0].numpy()  # show first data
data = train_set.data[0].clone()
data[data>0] = 1                        # convert to 1 if pixel greater than 0
data = data.numpy()
#plt.imshow(train_set.data[0].reshape(28,28), cmap='gray')
#plt.axis('off') # conceal scale
#plt.show()

#%% construct model(使用類別定義模型)
# must inheirt nn.Module
# include two method: __init__, forwwrd
# __init__: declare object of neural layer
# forward: contact each layer
class Net(nn.Module):
    def __init__(self):
        super.__init__()
        self.fc1 = nn.Linear(28*28, 256) # fully connected 1, upper case
        self.dropout1 = nn.Dropout(0.2)  # drop layer, upper case
        self.fc2 = nn.Linear(256, 10)    # fully connected 2, upper case
    
    def forward(self, x):
        x = torch.flatten(x, 1) # reference: https://pytorch.org/docs/stable/generated/torch.flatten.html?highlight=flatten#torch.flatten
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        #x = self.dropout2(x)
        #output = F.softmax(x, dim=1) # lower case, 使用CrossEntropy時不可加softmax
        #return output
        return x

model = Net().to(device)
#%% start train

epochs = 2
learning_rate = 0.1

# construct dataloader
train_loader = DataLoader(train_set, batch_size=batch_size)

# optimizer
optim = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# loss function
criterion = nn.CrossEntropyLoss()


model.train() # model enter train, all layer will operate
loss_list = []

for epoch in range(1, epochs+1):
    print(f"Epoch: {epoch}/{epochs}")
    print('-' * 10)
    for index, (data, target) in enumerate(train_loader): # index from 1 to 100
        data, target = data.to(device), target.to(device) # convert to GPU
        
        optim.zero_grad()                 # reset gradient
        output = model(data)
        loss = criterion(output, target)  # calculate loss
        loss.backward()                   # back propagation
        optim.step()                      # update weight
        
        loss_list.append(loss.item())  # save loss
        
        # every 10 times(6000 data) show loss value
        if (index % 10) == 0:
            batch = index * len(data)
            data_count = len(train_loader.dataset)
            print(f"[{batch:5d} / {data_count}]  Loss: {loss.item()}")

#%% visualization
#plt.plot(range(1, len(loss_list)+1), loss_list)
#plt.show()
   
#%% test
# construct dataloader
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

model.eval() # model enter test, drop layer will not be operate
test_loss = 0
correct_1 = 0
correct_2 = 0

with torch.no_grad():
    for data, target in test_loader:
        data ,target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(target, output).item() # sum batch loss
        #print(target.size())
        #print(output.size())
                               
        # prediction
        # line:1-2 and line:3-4 are equivalent
        # method (1) make sure two array(pred and target) has same dimension
        # method (2) use python operationto compare
        # 兩個方法等價, 但第一個可以確保維度一樣, 較不易出錯, 第二個使用python的邏輯運算
        pred = output.argmax(dim=1, keepdim=True) # shape = [n_sample(600), prediction(10)]
        correct_1 += pred.eq(target.view_as(pred)).sum().item()
        #y_pred = output.argmax(dim=1)            # shape = [prediction(600)]
        #correct_2 += torch.sum(y_pred==target).item()

# average loss
test_loss /= len(test_loader.dataset)
print(f"Average Loss = {test_loss}, Accuracy = {correct_1} / {len(test_loader.dataset)}")
print(f"Average Loss = {test_loss}, Accuracy = {correct_1} / {len(test_loader.dataset)}")

