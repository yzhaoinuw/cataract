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
):

    sample_loss = np.zeros(len(df_labels))
    test_losses = []
    baseline_losses = []
    for i in range(runs):
        if i % 100 == 0:
            print(f"experiment {i}.")
        exp = Experiment(df_features, df_labels, test_size=0.4, h1=h1, dropout=0)
        exp.run_train(epochs=epochs)
        test_loss = exp.run_test()
        baseline_loss = exp.compute_baseline_loss()
        baseline_losses.append(baseline_loss)
        sample_loss += exp.compute_test_sample_loss()
        test_losses.append(test_loss)

    baseline_losses = np.array(baseline_losses)
    test_losses = np.array(test_losses)
    return {
        "test_losses": test_losses,
        "sample_loss": sample_loss,
        "baseline_losses": baseline_losses,
    }


#%%
DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_labels = pd.read_excel(DATA_PATH + label_file)

# dropping columns
# df_features = df_features.drop("Age", axis=1)
# df_features = df_features[["EPP/LT", "ACD_pre (mm)"]]
losses = run_experiment(df_features, df_labels)

#%%
"""
plt.figure(figsize=(10, 5))
plt.bar([str(k) for k in range(1, len(df_labels)+1)], sample_loss/runs)
plt.title(f"Average MAE per Sample (from {runs} runs)")
plt.xlabel("Sample ID")
plt.ylabel("Average Loss")

#plt.title("Training and Test Loss")
#plt.plot(list(range(1, EPOCHS+1)), test_losses, label="test")
#plt.plot(list(range(1, EPOCHS+1)), train_losses, label="train")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
#plt.grid()
#plt.legend()
plt.show()
"""
