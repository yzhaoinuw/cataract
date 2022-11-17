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
feature_file = "features_processed.xlsx"

# df_features = pd.read_excel(DATA_PATH + feature_file)
# df_labels = pd.read_excel(DATA_PATH + label_file)
df_features = pd.read_excel(DATA_PATH + feature_file)
df_labels = df_features["ELP"].to_frame()

# df_labels = (df_labels - 4) * 1000
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

#%%
train_losses = exp.get_train_losses()
test_losses = exp.get_test_losses()

skip_epochs = 50
plt.title("Training and Test Loss")
plt.plot(
    list(range(skip_epochs + 1, epochs + 1)), test_losses[skip_epochs:], label="test"
)
plt.plot(
    list(range(skip_epochs + 1, epochs + 1)), train_losses[skip_epochs:], label="train"
)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.ylim((0, 1))
plt.grid()
plt.legend()
plt.show()
#%%
model = exp.model
device = exp.device
X_train, X_test = exp.X_train.to(device), exp.X_test.to(device)
y_train, y_test = exp.y_train.to(device), exp.y_test.to(device)
