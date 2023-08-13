# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:32:09 2023

@author: Chen ZE
"""

# Chap3.3.3 Gradient Accumulation(梯度累加)
# gradient will accumulate
# so if want to calculate backword repeat, we must reset gradient: .grad.zero_()

import torch as t

x = t.tensor(5.0, requires_grad=True)
y = x **3 # y = x^3
z = y **2 # z = y^2 = x^6

y.backward(retain_graph=True) # first backword
print(f"first backword, gradient of x = {x.grad}") # 75, if not reset
x.grad.zero_()

y.backward(retain_graph=True) # second backword
print(f"second backword, gradient of x = {x.grad}") # 150 = 75 + 75, if not reset
x.grad.zero_()

y.backward(retain_graph=True) # second backword
print(f"third backword, gradient of x = {x.grad}") # 225 = 150 + 75, if not reset
x.grad.zero_()


# multiple variable derivative
z.backward(retain_graph=True)
print(f"gradient of x = {x.grad}") # 18750
#print(f"gradient of y = {y.grad}") # 250



