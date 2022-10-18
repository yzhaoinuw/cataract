# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:23:49 2022

@author: Yue
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, h1=128, dropout=0.5):
        super(MLP, self).__init__()
        # self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, h1)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(h1, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h1 = self.relu(self.l1(x))
        output = self.dropout(self.l2(h1))
        return output


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
  """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            # print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()
