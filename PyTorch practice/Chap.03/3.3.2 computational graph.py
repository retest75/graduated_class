# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:57:04 2023

@author: Chen ZE
"""

# Chap 3.3.2 Computational Graph
# if you want to derivative some variable
# requires_grad must be "True"
# for example: Loss = CrossEntropy(y, z), where z = w*x + b
# computational graph will be destroy after backward(), so you can't operate second backward()
# if want to operate second backword, retain_graph must be True

import torch

x = torch.ones(5)  
y = torch.zeros(3)  
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
#print(z.requires_grad) # True, because z is function of w and b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

#print('gradient function of z：', z.grad_fn)  # <AddBackward>
#print('gradient function of Loss：', loss.grad_fn) # <BinaryCrossEntropyWithLogitsBackward>

# backpropagation
loss.backward()
#loss.back(retain_graph=True)
print(w.grad)         # gradient of w
print(b.grad)         # gradient of b

#loss.backward() #can't not operate

