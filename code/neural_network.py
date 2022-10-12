# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:23:49 2022

@author: Yue
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            # nn.ReLU()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
  """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            #print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()
