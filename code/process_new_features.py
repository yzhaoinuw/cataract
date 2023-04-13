# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:40:37 2022

@author: Yue

add headers to Edu's new features table
"""

import pandas as pd

from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "../data/"
feature_file = "Features7.xlsx"

df = pd.read_excel(DATA_PATH + feature_file, header=None)
col_names = [
    "Sex",  # 1
    "Laterality",
    "AgeAtTimeOfOperationyear",
    "IOLModel",
    "IOLPowerInsertedD",  # 5
    "AxialLengthmm",
    "PreopK1",
    "PreopK1Axis",
    "PreopK2",
    "PreopK2Axis",  # 10
    "Sphere",
    "Cyl",
    "SphericalEquiv",
    "NumberOfDaysPostOpScan",
    "PupilSize",  # 15
    "RAC",
    "CT",
    "ACD",
    "LT",
    "VCD",  # 20
    "AL",
    "ALNotCorrected",
    "StdALNonCorrectedEyes",
    "MedRACEyes",
    "MedRPCEyes",  # 25
    "MedRALEyes",
    "MedRPLEyes",
    "MedRACEyesDiam2",
    "MedRPCEyesDiam2",
    "MedRALEyesDiam2",  # 30
    "MedRPLEyesDiam2",
    "RAC3D",
    "RPC3D",
    "RAL3D",
    "RPL3D",  # 35
    "RAC3DDiam2",
    "RPC3DDiam2",
    "RAL3DDiam2",
    "RPL3DDiam2",
    "CTPostEyes",  # 40
    "IOLTEyes",
    "VCDPostEyes",
    "ALPostEyes",
    "ALNonCorrectedPostEyes",
    "LP",  # 45
    "ALPost",
    "SpherePost",
    "CylinderPost",
    "SphericalEquivPost",
]
df.columns = col_names
cat_cols = ["Sex", "Laterality", "IOLModel"]
df = df.dropna(subset=cat_cols)
df = df.astype({"IOLModel": int})
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
        pd.DataFrame(codes, columns=feature_names, index=df.index).astype(int),
    ],
    axis=1,
)

#%%
SAVE_PATH = DATA_PATH + "features7_processed.xlsx"
df_new.to_excel(SAVE_PATH, index=False)
