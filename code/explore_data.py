# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:53:52 2022

@author: Yue
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

DATA_PATH = "../data/"
FIGURE_PATH = "../figure/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"
feature_file = "features_processed.xlsx"
feature_file = "features_processed_dec.xlsx"

# df_features = pd.read_excel(DATA_PATH + feature_file)
# df_label = pd.read_excel(DATA_PATH + label_file)

# new data
df_features = pd.read_excel(DATA_PATH + feature_file)
df_features = df_features.dropna()
# df_features = df_features[df_features["IOLModel_1"] == 1]
# df_features = df_features.drop("IOLModel_1", axis=1)
df_labels = df_features["LP"].to_frame()
df_features = df_features.drop("LP", axis=1)

# try some transformation
# df_labels = (df_labels - 4) * 1

#%%

X = df_features
y = df_labels.values.flatten()
# PCA
# pca = PCA(n_components=5)
# pca.fit(df_features.values)

model = Ridge()
model.fit(X, y)

coefs = pd.DataFrame(model.coef_, columns=["Coefficients"], index=df_features.columns)

coefs.plot(kind="barh", figsize=(9, 7))
plt.title("Ridge model")
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
#%%
"""
# replicate Edu's work
X = df_features.values
y = df_labels.values
y_pred = 8.497 * df_features["EPP/LT"] + 0.25 * df_features["ACD_pre (mm)"]
mae = abs(y_pred - df_labels["LP"]).mean()
me = (y_pred - df_labels["LP"]).mean()

#%%
df_features["ELP"] = y
feature_names = df_features.columns
corr = df_features.corr()
# Fill diagonal and upper half with NaNs
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan

fig = plt.figure()
ax = plt.gca()
img = ax.matshow(corr)
fig.colorbar(img)
ax.set_xticks(np.arange(len(feature_names)))
ax.set_xticklabels(feature_names)
ax.set_yticks(np.arange(len(feature_names)))
ax.set_yticklabels(feature_names)

ax.xaxis.set_ticks_position("bottom")

# Set ticks on both sides of axes on
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
# Rotate and align bottom ticklabels
plt.setp(
    [tick.label1 for tick in ax.xaxis.get_major_ticks()],
    rotation=45,
    ha="right",
    va="center",
    rotation_mode="anchor",
)
ax.set_title("Eye Features Correlation")
ax.tick_params(axis="both", labelsize=5)
# fig.tight_layout()
# plt.savefig(FIGURE_PATH+'eye_features_correlation.png', dpi=300, bbox_inches="tight")
plt.show()
"""
