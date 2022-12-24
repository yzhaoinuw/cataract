# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:58:13 2022

@author: Yue
"""

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dataset import Dataset
from neural_network import MLP


class Experiment:
    def __init__(
        self,
        df_features,
        df_labels,
        test_size=0.4,
        split_seed=None,
        h1=64,
        dropout=0,
        epochs=50,
        normalization=True,
        batchnorm=False,
    ):
        use_cuda = torch.cuda.is_available()
        self.data_size = len(df_labels)
        self.test_size = test_size
        self.split_seed = split_seed
        self.normalization = normalization
        self.batchnorm = batchnorm
        self.h1 = h1
        self.dropout = dropout
        self.epochs = epochs
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.loss_function = nn.MSELoss()
        self.loss_function_plot = nn.L1Loss(reduction="none")
        self._initialize(df_features, df_labels)

    def _initialize(self, df_features, df_labels):
        df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
            df_features,
            df_labels,
            test_size=self.test_size,
            random_state=self.split_seed,
        )
        self.test_indices = df_X_test.index
        X_train, X_test = df_X_train.values, df_X_test.values
        y_train, y_test = df_y_train.values, df_y_test.values
        if self.normalization:
            scaler_x = StandardScaler()
            X_train = scaler_x.fit_transform(X_train)
            X_test = scaler_x.transform(X_test)

        self.X_train = torch.from_numpy(X_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).float()
        input_size = self.X_train.shape[1]
        # Initialize the MLP
        self.model = MLP(
            input_size=input_size,
            h1=self.h1,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_losses = []
        self.test_losses = []

    def run_train(self, verbose=False, run_test=True, batch_size=4):
        train_set = Dataset(self.X_train, self.y_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        self.train_losses = []
        self.test_losses = []
        for epoch in range(1, self.epochs + 1):
            if run_test:
                self.test_losses.append(self.run_test())
            current_loss = 0.0
            self.model.train()
            for i, data in enumerate(train_loader, 0):
                # Get and prepare inputs
                inputs, targets = data
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
                current_loss += torch.sum(
                    self.loss_function_plot(outputs, targets)
                ).item()

            current_loss /= len(train_set)
            self.train_losses.append(current_loss)
            if verbose:
                if epoch % 10 == 0:
                    print(f"finished epoch {epoch}")
                    print(f"training loss: {current_loss}")
                    print("")

    @torch.no_grad()
    def run_test(self, X_test=None, y_test=None, take_mean=True, verbose=False):
        self.model.eval()
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        self.model.eval()
        y_pred = self.model(X_test)
        test_loss = (
            self.loss_function_plot(y_pred, y_test).detach().cpu().numpy().squeeze()
        )
        if take_mean:
            test_loss = np.mean(test_loss).item()
        if verbose:
            print(f"test loss: {test_loss}")
        return test_loss

    def get_train_losses(self) -> np.ndarray:
        return np.array(self.train_losses)

    def get_test_losses(self) -> np.ndarray:
        return np.array(self.test_losses)

    def compute_test_sample_loss(self):
        test_sample_loss = np.zeros(self.data_size)
        test_loss = self.run_test(take_mean=False)
        test_sample_loss[self.test_indices] += test_loss
        return test_sample_loss

    def compute_baseline_loss(self, take_mean=True):
        baseline_pred = (
            torch.mean(self.y_train).repeat(self.y_test.shape).detach().cpu()
        )
        baseline_loss = (
            self.loss_function_plot(baseline_pred, self.y_test).detach().cpu()
        )
        if take_mean:
            baseline_loss = torch.mean(baseline_loss).item()
        return baseline_loss
