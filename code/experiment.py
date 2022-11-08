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


class Experiment:
    def __init__(self, df_features, df_labels, test_size=0.4, split_seed=None, h1=256, dropout=0):
        use_cuda = torch.cuda.is_available()
        self.data_size = len(df_labels)
        self.test_size = test_size
        self.split_seed = split_seed
        self.h1 = h1
        self.dropout = dropout
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.loss_function = nn.MSELoss()
        self.loss_function_plot = nn.L1Loss(reduction="none")
        self._initialize(df_features, df_labels)

    def _initialize(self, df_features, df_labels):
        X_train, X_test, y_train, y_test = train_test_split(
            df_features, df_labels, test_size=self.test_size, random_state=self.split_seed
        )

        self.test_indices = X_test.index
        X_train = torch.from_numpy(X_train.values).float()
        y_train = torch.from_numpy(y_train.values).float()
        self.X_test = torch.from_numpy(X_test.values).float()
        self.y_test = torch.from_numpy(y_test.values).float()
        self.train_set = Dataset(X_train, y_train)
        self.input_size = X_train.shape[1]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=4, shuffle=True
        )
        # Initialize the MLP
        self.model = MLP(input_size=self.input_size, h1=self.h1, dropout=self.dropout).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def run_train(self, epochs=50, verbose=False):
        train_losses = []
        for epoch in range(1, epochs + 1):
            current_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # Get and prepare inputs
                inputs, targets = data
                # inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.reshape((targets.shape[0], 1))
                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

                # Print statistics
                current_loss += torch.mean(self.loss_function_plot(outputs, targets)).item()
 
            train_losses.append(current_loss)
            if verbose:
                if epoch % 10 == 0:
                    print(f"finished epoch {epoch}")
                    print(f"training loss: {current_loss}")
                    print("")
        return train_losses
    
    @torch.no_grad()
    def run_test(self, X_test=None, y_test=None, take_mean=True, verbose=False):
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        y_pred = self.model(X_test)
        test_loss = self.loss_function_plot(y_pred, y_test).detach().cpu()
        if take_mean:
            test_loss = torch.mean(test_loss).item()
        if verbose:
            print(f"test loss: {test_loss}")
        return test_loss
        
    def get_test_sample_loss(self):
        test_sample_loss = torch.zeros(self.data_size)
        test_loss = self.run_test(take_mean=False)
        test_sample_loss[self.test_indices] += test_loss.squeeze()
        return test_sample_loss.numpy()