# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:53:34 2023

@author: Chen ZE
"""

# Chap3.4.2 Application of Fully Connected
# reference(Flatten):https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html?highlight=flatten#torch.nn.Flatten
# line 21-22: input must be 2-dimension, otherwise flatten will false
# line 23-22: numpy default float64, but pytorch default float32, must convert data type
# line 50-52: replace updating w and b in sequentially with model.parameters
# line 64: replace reset w and b in sequentially with model.zero_grad()

import torch
import numpy as np
import matplotlib.pyplot as plt

# produce sample with 1 feature, i.e. shape=[n, 1]
n = 100
x = np.linspace(0, 50, n) + np.random.uniform(-10, 10, n) # noise
x = torch.FloatTensor(x.reshape(n, 1)) 
y = np.linspace(0, 50, n) + np.random.uniform(-10, 10, n) # noise
y = torch.FloatTensor(y)

# construct model
model = torch.nn.Sequential(torch.nn.Linear(1, 1),
                            torch.nn.Flatten(0, -1)
                            )

# loss function
MSE = torch.nn.MSELoss(reduction='sum')


def train(x, y, epochs=2000, learning_rate=1e-6):
    w_list, b_list, loss_list = [], [], []
    
    # forward propagation
    for i in range(epochs):
        print(f"Epochs: {i+1}/{epochs}")
        print("-"*10)
        y_pred = model(x)
        loss = MSE(y_pred, y)
        
        # back propagation
        loss.backward()
        print(f"Loss: {loss.item()}")

        # update parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
                
        # record train result
        linear_layer = model[0] # torch.nn.Linear
        print(f"w = {linear_layer.weight.item()}")
        print(f"b = {linear_layer.bias.item()}")
        print()
        w_list.append(linear_layer.weight.item())
        b_list.append(linear_layer.bias.item())
        loss_list.append(loss.item())
        
        # reset gradient
        model.zero_grad()
    
    print(f"Best weight: {w_list[-1]}")
    print(f"Best bias: {b_list[-1]}")
    
    return w_list, b_list, loss_list

epochs = 20
w_list, b_list, loss_list = train(x, y, epochs=epochs)

# check by graph
y_pred = x * w_list[-1] + b_list[-1]
plt.scatter(x, y, label="data", color="cyan")
plt.plot(x, y_pred, label="prediction", color="red")
plt.legend()
plt.show()

# loss function
plt.plot(range(1, epochs+1), loss_list, label="loss")
plt.legend()
plt.show()
    
    
    


