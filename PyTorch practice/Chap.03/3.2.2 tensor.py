# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:37:17 2023

@author: Chen ZE
"""

# 3.2 Tensor Operation
# (1) create tensor
# (2) observe shape
# (3) operation
# (4) convert

import numpy as np
import torch as t

# -----create and basic operation
t1 = t.tensor([[1, 2],
               [3, 4]])

t2 = t.tensor([[1, 1],
               [1, 1]])

zeros = t.zeros((2, 3))
ones = t.ones((1, 2))
iden = t.tensor([[1, 0],
                 [0, 1]])
#print(ones)

# -----observe shape
shape = zeros.shape
shape1 = zeros.shape[1]
size = zeros.size
size1 = zeros.size(1)
#print(size)

# -----operation
a = t1 + t2
b = t1 - t2
c = t1 * t2
d = t1 / t2
#print(a)

# -----convert
n = np.array([[0, 1],
              [2, 3]])
#np_to_tensor = t.tensor(n)
np_to_tensor = t.from_numpy(n)
tensor_to_np = np_to_tensor.numpy()
print(tensor_to_np)

             


