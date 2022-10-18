# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:36:04 2022

@author: Yue
"""

import logging

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


def cross_validate(
    folds=6, epochs=50, save_model=False, model_name=None, verbose=False,
):
    logging.basicConfig()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    if model_name is not None:
        model_name = f"MLPRegressor_epoch{epochs}_{folds}folds"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    cv_losses = []
    kf = KFold(n_splits=folds)
    for i, (train_index, test_index) in enumerate(kf.split(X), 1):
        logging.info("Fold {}".format(i))
        logging.info(
            "TRAIN: {}. TEST: {}".format(
                ", ".join(map(str, train_index)), ", ".join(map(str, test_index))
            )
        )
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()

        # Initialize the MLP
        mlp = MLP(input_size=X_train.shape[1]).to(device)
        mlp.apply(reset_weights)

        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        loss_function_test = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
        train_losses = []
        test_losses = []

        for epoch in range(1, epochs + 1):
            logging.debug("Starting epoch {}".format(epoch))

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

            logging.debug("training loss: {}.".format(train_loss))
            train_losses.append(train_loss)

            with torch.no_grad():
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_pred = mlp(X_test)
                loss = loss_function_test(y_pred, y_test)
                test_loss = loss.item() / len(X_test)
                test_losses.append(test_loss)
                logging.debug("test loss: {}.".format(test_loss))
                logging.debug("")
        min_test_loss = np.array(test_losses).min()
        min_test_loss_epoch = np.array(test_losses).argmin()
        logging.info("min train loss: {}.".format(train_loss))
        logging.info(
            "min test loss: {} at epoch {}.".format(min_test_loss, min_test_loss_epoch)
        )
        logging.info("")
        cv_losses.append(min_test_loss)
    logging.info("average test loss: {}.".format(np.array(cv_losses).mean()))


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

FOLDS = 5
EPOCHS = 100
SAVE_MODEL = False
cross_validate(
    folds=FOLDS, epochs=EPOCHS, save_model=SAVE_MODEL,
)
