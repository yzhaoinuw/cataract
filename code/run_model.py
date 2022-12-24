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
label_file = "label.xlsx"
feature_file = "features_processed_dec.xlsx"

# df_labels = pd.read_excel(DATA_PATH + label_file)
df_features = pd.read_excel(DATA_PATH + feature_file)

use_features = [
    "AxialLengthmm",
    "RAC",
    "IOLModel_1",
    "IOLModel_2",
    "IOLModel_3",
    "Sex_1",
    "Sex_2",
    # Axial measurements, col 17 - 20
    # "CT",
    # "ACD",
    # "LT",
    # "VCD",
    # new AL
    # "AL",
    # crystalline lens params, set I, col 26-27
    # "MedRALEyes",
    # "MedRPLEyes",
    # crystalline lens params, set II, col 30-31
    # "MedRALEyesDiam2",
    # "MedRPLEyesDiam2",
    # crystalline lens params, set III, col 34-35
    # "RAL3D",
    # "RPL3D",
    # crystalline lens params, set IV, col 38-39
    # "RAL3DDiam2",
    # "RPL3DDiam2",
    # additional features
    # "PupilSize",
    # LP
    "LP",
]

df_features = df_features[use_features]
df_labels = df_features["LP"].to_frame()
df_features = df_features.drop("LP", axis=1)

# try some transformation
df_labels = (df_labels - 4) * 1

#%%
h1 = 32
epochs = 50
normalization = True
batchnorm = False
batch_size = 8

exp = Experiment(
    df_features,
    df_labels,
    test_size=0.4,
    h1=h1,
    epochs=epochs,
    dropout=0,
    normalization=normalization,
    batchnorm=batchnorm,
)
exp.run_train(batch_size=batch_size)

#%%
train_losses = exp.get_train_losses()
test_losses = exp.get_test_losses()
baseline_loss = exp.compute_baseline_loss()
sample_loss = exp.compute_test_sample_loss()

skip_epochs = 20
plt.title("Training and Test Loss")
plt.plot(
    list(range(skip_epochs + 1, epochs + 1)), test_losses[skip_epochs:], label="test"
)
plt.plot(
    list(range(skip_epochs + 1, epochs + 1)), train_losses[skip_epochs:], label="train"
)
plt.axhline(y=baseline_loss, color="r", linestyle="-", label="baseline loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.xlim(skip_epochs, epochs)
# plt.ylim((0, 0.2))
plt.grid()
plt.legend()
plt.show()
#%%
model = exp.model
device = exp.device
X_train, X_test = exp.X_train.to(device), exp.X_test.to(device)
y_train, y_test = exp.y_train.to(device), exp.y_test.to(device)
pred = model(X_test)
