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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from neural_network import MLP
from run_neural_network import run_neural_nextwork

DATA_PATH = "../data/"

feature_file = "features7_processed.xlsx"
df_features = pd.read_excel(DATA_PATH + feature_file)

# select subset of features
use_features = [
    #"AxialLengthmm",
    "RAC",
    "IOLModel_1",
    "IOLModel_2",
    "IOLModel_3",
    # "Sex_1",
    # "Sex_2",
    # Axial measurements, col 17 - 20
    "CT",
    "ACD",
    "LT",
    "VCD",
    # new AL
    "AL",
    # crystalline lens params, set I, col 26-27
    # "MedRALEyes",
    # "MedRPLEyes",
    # crystalline lens params, set II, col 30-31
    # "MedRALEyesDiam2",
    # "MedRPLEyesDiam2",
    # crystalline lens params, set III, col 34-35
    "RAL3D",
    "RPL3D",
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

X = df_features.values
y = df_labels.values.flatten()

#%%
@ignore_warnings(category=ConvergenceWarning)
def cross_validate(
    folds=5,
    epochs=100,
    model_name=None,
    verbose=False,
):
    logging.basicConfig()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    if model_name is not None:
        model_name = f"MLPRegressor_epoch{epochs}_{folds}folds"

    cv_losses = []
    cv_std = []
    kf = KFold(n_splits=folds, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X), 1):
        #logging.info("Fold {}".format(i))
        #logging.info(
        #    "TRAIN: {}. TEST: {}".format(
        #        ", ".join(map(str, train_index)), ", ".join(map(str, test_index))
        #    )
        #)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)
        
        #model = MLPRegressor(
        #    hidden_layer_sizes=(16,),
        #    max_iter=50,
            #early_stopping=True,
            #validation_fraction=0.2,
        #)
        model = Ridge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        absolute_errors = abs(y_pred-y_test)
        mae = absolute_errors.mean()
        std = absolute_errors.std()
        #train_losses, test_losses = run_neural_nextwork(X_train, X_test, y_train, y_test, epochs=50)
        #min_test_loss = np.array(test_losses).min()
        #min_test_loss_epoch = np.array(test_losses).argmin()
        #test_loss = test_losses[-1]
        #logging.info("min train loss: {}.".format(np.array(train_losses).min()))
        #logging.info(
        #    "min test loss: {} at epoch {}.".format(min_test_loss, min_test_loss_epoch)
        #)
        #logging.info("")
        cv_losses.append(mae)
        cv_std.append(std)
    #logging.info("test loss mean: {}.".format(np.array(cv_losses).mean()))
    #logging.info("test loss std: {}.".format(np.array(cv_losses).std()))
    return np.array(cv_losses).mean(), np.array(cv_std).mean()


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
EPOCHS = 50
losses = []
stds = []
for i in range(200):
    
    loss, std = cross_validate(
        folds=FOLDS,
        epochs=EPOCHS,
    )
    losses.append(loss)
    stds.append(std)
 

mae_mean = np.array(losses).mean()
mae_std = np.array(stds).mean()

print(f"MAE mean: {mae_mean}")
print(f"MAE std: {mae_std}")
