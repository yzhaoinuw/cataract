# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:43:21 2022

@author: Yue
"""

from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from experiment import Experiment

DATA_PATH = "../data/"

# previous data
# feature_file = "Tables_1_2_data.xlsx"
# label_file = "label.xlsx"
# df_features = pd.read_excel(DATA_PATH + feature_file)
# df_labels = pd.read_excel(DATA_PATH + label_file)

# new data
feature_file = "features_processed.xlsx"
df_features = pd.read_excel(DATA_PATH + feature_file)
df_labels = df_features["LP"].to_frame()
df_features = df_features.drop("LP", axis=1)

# try some transformation
df_labels = (df_labels - 4) * 1
# dropping columns
# df_features = df_features.drop("EPP/LT", axis=1)
# df_features = df_features[["EPP/LT", "ACD_pre (mm)"]]

#%%
runs = 1000
hidden_size = 32
epochs = 50
normalization = True
batch_size = 8
# permuting values in a column
# permute_col = "Age"
feature_loss = defaultdict(list)
for k in range(runs):
    if k % 100 == 0:
        print(f"experiment {k}.")
    exp = Experiment(
        df_features,
        df_labels,
        h1=hidden_size,
        epochs=epochs,
        normalization=normalization,
    )
    train_losses = exp.run_train(batch_size=batch_size)
    test_loss = exp.run_test()
    feature_loss["none"].append(test_loss)

    X_test = exp.X_test
    for i in range(len(df_features.columns)):
        row_inds = torch.randperm(X_test.shape[0])
        X = X_test.clone()
        X[:, i] = X_test[row_inds, i]
        feature_test_loss = exp.run_test(X_test=X)
        feature_loss[df_features.columns[i]].append(feature_test_loss)

for feature, losses in feature_loss.items():
    losses = np.array(losses)
    print(feature)
    print(f"mean: {losses.mean()}")
    print(f"std: {losses.std()}")
    print()
