# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:20:46 2023

@author: Chen ZE
"""

# Chap3.4.3 Application of Fully Connected(import optimizer)
# compare with 3.4.2
# line 39: increasing optimizer to control learning rate(Adam)
# line 52-56: use optimizer to control updating weight
# line 67-69: use optimizer to control gradient reset
# reference: https://pytorch.org/docs/stable/optim.html


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
    
    # optimizer
    optim = torch.nn.optim.Adam(model.parameters(), learning_rate)

    # forward propagation
    for i in range(epochs):
        print(f"Epochs: {i+1}/{epochs}")
        print("-"*10)
        y_pred = model(x)
        loss = MSE(y_pred, y)
        
        # back propagation
        loss.backward()
        print(f"Loss: {loss.item()}")

        # update parameters: use optimizer
        #with torch.no_grad():
        #    for param in model.parameters():
        #        param -= learning_rate * param.grad
        optim.step()
        
        # record train result
        linear_layer = model[0] # torch.nn.Linear
        print(f"w = {linear_layer.weight.item()}")
        print(f"b = {linear_layer.bias.item()}")
        print()
        w_list.append(linear_layer.weight.item())
        b_list.append(linear_layer.bias.item())
        loss_list.append(loss.item())
        
        # reset gradient: use optimizer
        #model.zero_grad()
        optim.zero_grad()
    
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