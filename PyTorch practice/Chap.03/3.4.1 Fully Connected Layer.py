# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:36:41 2023

@author: Chen ZE
"""

# Chap3.4.1 Fully Connected Layer
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# reference: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

import torch

input = torch.rand(128, 20)     # input size = [128, 20]
layer = torch.nn.Linear(20, 30) # input 20 neural, output 30 neural
output = layer(input)
print(output.size())            # output size = [128, 30]
