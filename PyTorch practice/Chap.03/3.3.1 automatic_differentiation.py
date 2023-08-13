# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:43:02 2023

@author: Chen ZE
"""
# Chap 3.3.1 Automatic Differentiation
# if you want to derivative some variable
# requires_grad must be "True"
# supposed y is a function of x, and x can be derivative
# then y can be derivative also
# that is you don't have to set requires_grad to be True

import torch as t

x = t.tensor(4., requires_grad=True)
y = x **2

print(f"value of y: {y}")
print(f"gradient function of y: {y.grad_fn}") # <PowBackward>
print(f"check if y can derivative: {y.requires_grad}")
y.backward() # backpropagation
print(f"gradient of x: {x.grad}")
