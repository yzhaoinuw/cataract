# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 01:14:34 2022

@author: Yue
"""

from dataset import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from neural_network import MLP


DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_label = pd.read_excel(DATA_PATH + label_file)

#df_features = df_features.drop("EPP/LT", axis=1)

#%%

# scaler = MinMaxScaler()
X = df_features.values
y = df_label.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

# scaler_X = StandardScaler().fit(X_train)
# scaler_y = StandardScaler().fit(y_train)
# scaling
# X_train = torch.from_numpy(scaler_X.transform(X_train)).float()
# X_test = torch.from_numpy(scaler_X.transform(X_test)).float()
# y_train = torch.from_numpy(scaler_y.transform(y_train)).float()
# y_test = torch.from_numpy(scaler_y.transform(y_test)).float()

# no scaling
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

train_set = Dataset(X_train, y_train)
test_set = Dataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set)

#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# Initialize the MLP
mlp = MLP(input_size=X_train.shape[1], h1=512).to(device)
EPOCHS = 400

# Define the loss function and optimizer
loss_function = nn.MSELoss()
loss_function_plot = nn.L1Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
train_losses = []
test_losses = []

for epoch in range(1, EPOCHS+1):
    current_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # Get and prepare inputs
        inputs, targets = data
        # inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.reshape((targets.shape[0], 1))
        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss_function_plot(outputs, targets).detach().item()

    current_loss /= len(train_set)
    train_losses.append(current_loss)

    with torch.set_grad_enabled(False):
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_pred = mlp(X_test)
        test_loss = loss_function_plot(y_pred, y_test).item()
        test_losses.append(test_loss)
        
    if epoch % 50 == 0:
        print(f"finished epoch {epoch}")
        print(f"training loss: {current_loss}")
        print(f"test loss: {test_loss}")
        print("")

#%%
plt.figure(figsize=(10, 5))
plt.title("Training and Test Loss")
plt.plot(list(range(1, EPOCHS+1))[30:], test_losses[30:], label="test")
plt.plot(list(range(1, EPOCHS+1))[30:], train_losses[30:], label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
