# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:44:52 2022

@author: Yue
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_label = pd.read_excel(DATA_PATH + label_file)

X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_label, test_size=0.4
)
