# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:26:06 2023

@author: Yue
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from neural_network import MLP

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def run_neural_nextwork(X_train, X_test, y_train, y_test, epochs=50):
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    
    # Initialize the MLP
    mlp = MLP(input_size=X_train.shape[1], h1=32, dropout=0).to(device)
    mlp.reset_weights()
    
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    loss_function_test = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters())
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
        train_loss = loss.item()
    
        logging.debug("training loss: {}.".format(train_loss))
        train_losses.append(train_loss)
    
        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = mlp(X_test)
            loss = loss_function_test(y_pred, y_test)
            test_loss = loss.item()
            test_losses.append(test_loss)
            logging.debug("test loss: {}.".format(test_loss))
            logging.debug("")
    return train_losses, test_losses