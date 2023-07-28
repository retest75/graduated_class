# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:07:17 2023

@author: Chen ZE
"""

# self-attention mechanism

import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Self_Attention_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.query_weight = nn.Linear(in_features, out_features, bias=False)
        self.key_weight = nn.Linear(in_features, out_features, bias=False)
        self.value_weight = nn.Linear(in_features, out_features, bias=False)
        self.ReLU = nn.ReLU(inplace=True)
        self.out_projection = nn.Linear(out_features, out_features)
    
    def forward(self, x):
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)
        
        attention_score = torch.matmul(key.t(), query)
        attention_score = self.ReLU(attention_score)
        attention_value = torch.matmul(value, attention_score)
        output = self.out_projection(attention_value)
        return output


