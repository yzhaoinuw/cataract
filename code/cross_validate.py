# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:36:04 2022

@author: Yue
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from sklearn.model_selection import KFold

from neural_network import MLP, reset_weights

DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_label = pd.read_excel(DATA_PATH + label_file)

df_features = df_features.drop("EPP/LT", axis=1)

X = df_features.values
y = df_label.values

#%%
FOLDS = 5
EPOCHS = 50
SAVE_MODEL = False
MODEL_NAME = f"MLPRegressor_epoch{EPOCHS}_{FOLDS}folds"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

cv_losses = []
kf = KFold(n_splits=FOLDS)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    # Initialize the MLP
    mlp = MLP().to(device)
    mlp.apply(reset_weights)

    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}")

        # Get and prepare inputs
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_train = y_train.reshape((y_train.shape[0], 1))
        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(X_train)

        # Compute loss
        loss = loss_function(outputs, y_train)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        train_loss = loss.item() / len(X_train)

        print(f"training loss: {train_loss}")
        train_losses.append(train_loss)

        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = mlp(X_test)
            loss = loss_function(y_pred, y_test)
            test_loss = loss.item() / len(X_test)
            test_losses.append(test_loss)
            print(f"test loss: {test_loss}")
            print("")
    cv_losses.append(np.array(test_losses).min())
"""     
    plt.figure(figsize=(10,5))
    plt.title("Training and Test Loss")
    plt.plot(test_losses,label="test")
    plt.plot(train_losses,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
"""
