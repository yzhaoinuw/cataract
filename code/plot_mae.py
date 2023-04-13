# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 17:19:53 2022

@author: Yue
"""

import pandas as pd


RESULTS_PATH = "../results/"
data_file = "MAE_RAC+new_AL+IOL_Model+Axial_Measurements.xlsx"

df = pd.read_excel(RESULTS_PATH + data_file)
#%%
sample_count = df.astype(bool).sum(axis=0)
mae_mean = df.mean(axis=0)
mae_mean = df.sum(axis=0)/sample_count

ax = mae_mean.plot(
    lw=2,
    colormap="jet",
    marker=".",
    markersize=10,
    title="Mean MAE over 1000 Experiments per Eye",
)
ax.grid(True)
ax.set_xlabel("Eye ID")
ax.set_ylabel("Mean MAE")
