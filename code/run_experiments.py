# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 01:14:34 2022

@author: Yue
"""

from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment import Experiment


def run_experiment(
    df_features,
    df_labels,
    runs=1000,
    epochs=200,
    h1=256,
    normalization=True,
    batch_size=8,
    batchnorm=False,
):

    sample_loss = np.zeros(len(df_labels))
    test_losses = []
    test_sample_losses = []
    baseline_losses = []
    for i in range(runs):
        if i % 100 == 0:
            print(f"experiment {i}.")
        exp = Experiment(
            df_features,
            df_labels,
            test_size=0.4,
            h1=h1,
            epochs=epochs,
            dropout=0,
            normalization=normalization,
            batchnorm=batchnorm,
        )
        exp.run_train(batch_size=batch_size)
        baseline_loss = exp.compute_baseline_loss()
        baseline_losses.append(baseline_loss)
        # sample_loss += exp.compute_test_sample_loss()
        # train_losses = exp.get_train_losses()
        test_losses.append(exp.get_test_losses()[-1])
        test_sample_losses.append(exp.run_test(take_mean=False))

    baseline_losses = np.array(baseline_losses)
    test_losses = np.array(test_losses)
    test_sample_losses = np.array(test_sample_losses)
    return {
        # "train_losses": train_losses,
        "test_losses": test_losses,
        # "sample_loss": sample_loss,
        "baseline_losses": baseline_losses,
        "test_sample_losses": test_sample_losses,
    }


#%%
if __name__ == "__main__":
    DATA_PATH = "../data/"

    # previous data
    # feature_file = "Tables_1_2_data.xlsx"
    # label_file = "label.xlsx"
    # df_features = pd.read_excel(DATA_PATH + feature_file)
    # df_labels = pd.read_excel(DATA_PATH + label_file)

    # new data
    feature_file = "features_processed_dec.xlsx"
    df_features = pd.read_excel(DATA_PATH + feature_file)

    # drop features
    # df_features = df_features.drop("IOLPowerInsertedD", axis=1)

    # select subset of features
    use_features = [
        # "AxialLengthmm",
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
        # "RAL3D",
        # "RPL3D",
        # crystalline lens params, set IV, col 38-39
        # "RAL3DDiam2",
        # "RPL3DDiam2",
        # additional features
        # "PupilSize",
        # LP
        "LP",
    ]

    # Edu's benchmark
    # use_features = ["RAC", "AxialLengthmm", "IOLModel_1", "IOLModel_2", "IOLModel_3"]
    df_features = df_features[use_features]
    # df_features = df_features[df_features["IOLModel_1"] == 1]
    # df_features = df_features.drop("IOLModel_1", axis=1)
    df_labels = df_features["LP"].to_frame()
    df_features = df_features.drop("LP", axis=1)

    # try some transformation
    df_labels = (df_labels - 4) * 1

    # dropping columns
    # df_features = df_features.drop("Age", axis=1)
    # df_features = df_features[["EPP/LT", "ACD_pre (mm)"]]
    runs = 1000
    EPOCHS = 50
    hidden_size = 32
    batch_size = 8
    normalization = True
    losses = run_experiment(
        df_features,
        df_labels,
        runs=runs,
        h1=hidden_size,
        epochs=EPOCHS,
        normalization=normalization,
        batch_size=batch_size,
    )

    #%%

    test_losses = losses["test_losses"]
    baseline_losses = losses["baseline_losses"]
    # sample_loss = losses["sample_loss"]
    test_sample_losses = losses["test_sample_losses"]
    print(f"hidden_size: {hidden_size}")
    print(f"epochs: {EPOCHS}")
    print(f"test loss mean: {test_losses.mean()}")
    print(f"test loss std: {test_losses.std()}")
