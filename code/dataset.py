# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:02:43 2022

@author: Yue
"""

import torch


class Dataset(torch.utils.data.Dataset):
    """
    Prepare the Boston dataset for regression
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
