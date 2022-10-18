# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 01:14:34 2022

@author: Yue
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment import Experiment


DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_label = pd.read_excel(DATA_PATH + label_file)

# df_features = df_features.drop("EPP/LT", axis=1)
# df_features = df_features[["EPP/LT", "ACD_pre (mm)"]]
X = df_features.values
y = df_label.values

test_losses = []
for i in range(100):
    if i % 10 == 0:
        print(f"experiment {i}.")
    exp = Experiment(X, y, test_size=0.4, h1=512, dropout=0)
    test_loss = exp.run_experiement(epochs=100)
    # if test_loss > 1:
    #    break
    test_losses.append(test_loss)

#%%
test_losses = np.array(test_losses)

"""
plt.figure(figsize=(10, 5))
plt.title("Training and Test Loss")
plt.plot(list(range(1, EPOCHS+1)), test_losses, label="test")
plt.plot(list(range(1, EPOCHS+1)), train_losses, label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
"""
