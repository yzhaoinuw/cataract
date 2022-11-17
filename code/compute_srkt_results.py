# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:29:27 2022

@author: Yue
"""

import numpy as np
import pandas as pd

from SRKT_estimation import compute_ELP

DATA_PATH = "../data/"
feature_file = "features_new.xlsx"

df = pd.read_excel(DATA_PATH + feature_file, header=None)
col_names = [
    "Sex",
    "Laterality",
    "AgeAtTimeOfOperationyear",
    "IOLModel",
    "IOLPowerInsertedD",
    "AxialLengthmm",
    "PreopK1",
    "PreopK1Axis",
    "PreopK2",
    "PreopK2Axis",
    "Sphere",
    "Cyl",
    "SphericalEquiv",
    "NumberOfDaysPostOpScan",
    "RAC",
    "ELP",
]
df.columns = col_names
cat_cols = ["Sex", "Laterality", "IOLModel"]
for col in cat_cols:
    df[col] = df[col].astype(str)

#%%
RESULTS_PATH = "../results/"
ELPs = [
    compute_ELP(IOL_model, AL, RAC)
    for IOL_model, AL, RAC in zip(df["IOLModel"], df["AxialLengthmm"], df["RAC"])
]
ELPs = np.array(ELPs)
labels = df["ELP"].to_numpy()
MAE = abs(ELPs - labels).mean()

df_elp = pd.DataFrame(labels, columns=["LP"])
df_elp["ELP"] = ELPs
df_elp.to_excel(RESULTS_PATH + "SRKT_estimation.xlsx")
