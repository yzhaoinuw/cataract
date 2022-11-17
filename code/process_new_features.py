# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:40:37 2022

@author: Yue

add headers to Edu's new features table
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

from experiment import Experiment


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
    "LP",
]
df.columns = col_names

cat_cols = ["Sex", "Laterality", "IOLModel"]
for col in cat_cols:
    df[col] = df[col].astype(str)

#%%
enc = OneHotEncoder()
df_cat = df.select_dtypes("object")

enc.fit(df_cat)

codes = enc.transform(df_cat).toarray()
feature_names = enc.get_feature_names_out(cat_cols)

df_new = pd.concat(
    [
        df.select_dtypes(exclude="object"),
        pd.DataFrame(codes, columns=feature_names).astype(int),
    ],
    axis=1,
)

#%%
SAVE_PATH = DATA_PATH + "features_processed.xlsx"
# df_new.to_excel(SAVE_PATH, index=False)
