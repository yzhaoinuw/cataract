# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:58:13 2022

@author: Yue
"""

import torch
from torch import nn
from sklearn.model_selection import train_test_split

from dataset import Dataset
from neural_network import MLP

class Experiment():
    def __init__(self, X, y, test_size=0.4, split_seed=None, h1=128):
        use_cuda = torch.cuda.is_available()
        self.test_size = test_size
        self.split_seed = split_seed
        self.h1 = h1 
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.initialize(X, y)
        
    def initialize(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.split_seed)
        self.input_size = X_train.shape[1]
        # no scaling
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).float()
        self.train_set = Dataset(X_train, y_train)
        self.input_size = X_train.shape[1]
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=4, shuffle=True)
        # Initialize the MLP
        self.mlp = MLP(input_size=self.input_size, h1=self.h1).to(self.device)

    def run_experiement(self, epochs=50, verbose=False):

        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        loss_function_plot = nn.L1Loss(reduction="sum")
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-4)
        train_losses = []
        test_losses = []
        
        for epoch in range(1, epochs+1):
            
            with torch.no_grad():
                X_test, y_test = self.X_test.to(self.device), self.y_test.to(self.device)
                y_pred = self.mlp(X_test)
                test_loss = loss_function_plot(y_pred, y_test).item()
                test_loss /= y_test.shape[0]
                test_losses.append(test_loss)
                
            current_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
        
                # Get and prepare inputs
                inputs, targets = data
                # inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.reshape((targets.shape[0], 1))
                # Zero the gradients
                optimizer.zero_grad()
        
                # Perform forward pass
                outputs = self.mlp(inputs)
        
                # Compute loss
                loss = loss_function(outputs, targets)
        
                # Perform backward pass
                loss.backward()
        
                # Perform optimization
                optimizer.step()
        
                # Print statistics
                current_loss += loss_function_plot(outputs, targets).detach().item()
            current_loss /= len(self.train_set)
            train_losses.append(current_loss)
            
            if verbose:
                if epoch % 10 == 0:
                    print(f"finished epoch {epoch}")
                    print(f"training loss: {current_loss}")
                    print(f"test loss: {test_loss}")
                    print("")
        return test_losses[-1]