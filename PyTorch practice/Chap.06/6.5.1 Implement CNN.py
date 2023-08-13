# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:20:36 2023

@author: Chen ZE
"""

# Chap06.5.1 Implement CNN
# nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
# model: conv2d -> pooling -> conv2d -> pooling -> fc
# reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% construct model
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
                      nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                      nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                      nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                      nn.BatchNorm2d(32),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)   # output shape = [batch, c, h, w]=[batch, 16, 14, 14]
        x = self.layer2(x)   # output shape = [batch, c, h, w]=]batch, 32, 7, 7]
        x = torch.flatten(x, 1) # output shape = [batch, 32*7*7]
        x = self.fc(x)       # output shape = [batch, 10]
        output = F.log_softmax(x, dim=1)
        return output
        
model = ConvNet().to(device)

#%%
path_train = "E:/graduate computer/PyTorch/dataset/MNIST/train" # save dataset
path_test = "E:/graduate computer/PyTorch/dataset/MNIST/test"
train_set = MNIST(path_train, train=True, download=True, transform=transforms.ToTensor())
test_set = MNIST(path_test, train=False, download=True, transform=transforms.ToTensor())
#print(f"shape of train dataset: {train_set.data.shape}") # [60000, 28, 28], 60000筆資料
#print(f"shape of test dataaset: {test_set.data.shape}")  # [10000, 28, 28], 10000筆資料

# show data
#print(train_set.targets[:10])          # first 10 ground truth
#first_data = train_set.data[0].numpy()  # show first data
#data = train_set.data[0].clone()
#data[data>0] = 1                        # convert to 1 if pixel greater than 0
#data = data.numpy()
#plt.imshow(train_set.data[0].reshape(28,28), cmap='gray')
#plt.axis('off') # conceal scale
#plt.show()

#%%
epochs = 10
batch_size = 1000
learning_rate = 0.1
train_dataloader = DataLoader(train_set, batch_size=batch_size)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_list = []

model.train()
for epoch in range(1, epochs+1):
    print(f"{epoch}/{epochs}\n" + "-"*10)
    for index, (data, target) in enumerate(train_dataloader):
        data ,target = data.to(device), target.to(device)
        #print(target.size())
        #print("target=", target.dtype)
        #print("data=", data.dtype)

        optim.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optim.step()
        
        if ( (index+1) % 10 == 0):
            loss_list.append(loss.item())
            batch = (index+1) * len(data)
            print(f"{batch:5d}/{len(train_dataloader.dataset)}  Loss = {loss.item()}")

#%% visualization loss
#plt.plot(range(1, 61), loss_list, label="Loss")
#plt.legend()
#plt.show()
        
#%% validation

test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
test_loss = 0
correct1 = 0  # check for method 1
correct2 = 0  # check for method 2

model.eval()
with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        test_loss += F.nll_loss(output, target).item()
        
        # calculate correct number(1)
        _, y_pred1 = torch.max(output.data, 1)
        correct1 += (y_pred1 == target).sum().item()
        
        # calculate correct number(2)
        y_pred2 = output.argmax(dim=1)
        correct2 += torch.sum(y_pred1==target).item()

test_loss /= len(test_dataloader.dataset) # average loss

print(f"Average Loss: {test_loss:1.5f}")
#print(correct1)
#print(correct2)
print(f"Accuracy(method 1): {correct1}/{len(test_dataloader.dataset)}")
print(f"Accuracy(method 1): {correct1/len(test_dataloader.dataset)}")

print(f"Accuracy(method 2): {correct2}/{len(test_dataloader.dataset)}")
print(f"Accuracy(method 2): {correct2/len(test_dataloader.dataset)}")

#%% save model
#torch.save(model, "number_detection.pt")

#%% test

img_path = "E:/graduate computer/PyTorch/Chap.06/7.png"
img = Image.open(img_path)

transforms = transforms.Compose([transforms.Resize((28, 28)),
                                 transforms.Grayscale(),
                                 transforms.ToTensor()])
img = transforms(img)
img = torch.FloatTensor(1-img).to(device)
test_data = img.resize(1, 1, 28, 28)
output = torch.softmax(model(test_data), 1)
y_pred = np.argmax(output.detach().cpu().numpy())
for i in range(10):
    if y_pred == i:
        print(f"the number is {y_pred}")
#print(y_pred)



        
       
        

