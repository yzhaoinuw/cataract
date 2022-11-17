# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:53:36 2022

@author: Yue
"""


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

from experiment import Experiment


DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_labels = pd.read_excel(DATA_PATH + label_file)

df_labels = (df_labels - 4) * 1000
#%%
h1 = 256
epochs = 200
normalization = False

exp = Experiment(
    df_features,
    df_labels,
    test_size=0.4,
    h1=h1,
    epochs=epochs,
    dropout=0,
    normalization=normalization,
)
exp.run_train()
