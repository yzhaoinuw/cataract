# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:23:49 2022

@author: Yue
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, h1=32, dropout=0, batchnorm=False):
        super(MLP, self).__init__()
        l1 = nn.Linear(input_size, h1)
        relu = nn.LeakyReLU()
        l1_bn = nn.BatchNorm1d(h1)
        dropout = nn.Dropout(p=dropout)

        layers = [l1, relu, dropout]
        if batchnorm:
            layers.append(l1_bn)

        self.h1 = nn.Sequential(*layers)
        self.l2 = nn.Linear(h1, 1)

    def reset_weights(self):
        """
        Try resetting model weights to avoid
        weight leakage.
        """
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                # print(f"Reset trainable parameters of layer = {layer}")
                layer.reset_parameters()

    def forward(self, x):
        output = self.l2(self.h1(x))
        return output
    
class Linear(nn.Module):
    def __init__(self, input_size):
        super(Linear, self).__init__()
        self.l1 = nn.Linear(input_size, 1)
        self.intercept = nn.Parameter(torch.rand(1))
    
    def forward(self, x):
        return self.l1(x)
