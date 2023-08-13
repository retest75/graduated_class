# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:24:41 2023

@author: Chen ZE
"""

# Chap3.3.4 Simple Linear Regression by PyTorch
# use automatic derivative

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

def train(X, y, epochs=100, learning_rate=0.0001):
    w_list, b_list, loss_list = [], [], []
    
    # initialized weight and bias(常態分配的隨機亂數)
    w = torch.randn(1, requires_grad=True, dtype=torch.float)
    #w.to('cuda') # move to GPU
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    #b.to('cuda') # move to GPU
    
    # forward propagation
    for i in range(epochs):
        y_pred = w * X + b
        
        loss = torch.mean((y_pred - y)**2) # use MSE as loss function
        
        # backpropagation
        loss.backward()
        
        # update weigth and bias
        # when update weight, we shouldn't let parameter to participate gradient descent
        # because it may be affect gradient when operate backward propagation
        with torch.no_grad(): # 設定更新權重時不參與梯度下降
            w -= learning_rate * w.grad # w = w - eta*dw
            b -= learning_rate * b.grad # b = b - eta*db
        
        w_np = w.detach().numpy() # convert w to numpy
        b_np = b.detach().numpy() # convert b to numpy
        w_list.append(w_np)
        b_list.append(b_np)
        loss_list.append(loss.item()) # convert loss to constant
        
        # reset gradient
        w.grad.zero_()
        b.grad.zero_()
        
    return w_list, b_list, loss_list

# produce 100 data from 1 to 50
x = np.linspace(0, 50, 100) + np.random.uniform(-10, 10, 100) # noise
y = np.linspace(0, 50, 100) + np.random.uniform(-10, 10, 100) # noise
epochs = 50

w_list, b_list, loss_list = train(torch.tensor(x), torch.tensor(y), epochs=epochs)
print(f"best weight: {w_list[-1]}, best bias: {b_list[-1]}")

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

end = time.time()
print(f"operate time: {end - start}")

        
        