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
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_labels = pd.read_excel(DATA_PATH + label_file)

# dropping columns
# df_features = df_features.drop("EPP/LT", axis=1)
# df_features = df_features[["EPP/LT", "ACD_pre (mm)"]]

# permuting values in a column
# permute_col = "Age"
feature_loss = defaultdict(list)
for k in range(1000):
    if k % 100 == 0:
        print(f"experiment {k}.")
    exp = Experiment(df_features, df_labels, test_size=0.4, dropout=0)
    train_losses = exp.run_train(epochs=200)
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
