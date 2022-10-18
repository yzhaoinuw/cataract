# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:53:52 2022

@author: Yue
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_label = pd.read_excel(DATA_PATH + label_file)

# df_features = df_features.drop("EPP/LT", axis=1)

X = df_features.values
y = df_label.values

#%%
# replicate Edu's work
y_pred = 8.497 * df_features["EPP/LT"] + 0.25 * df_features["ACD_pre (mm)"]
mae = abs(y_pred - df_label["LP"]).mean()
me = (y_pred - df_label["LP"]).mean()
