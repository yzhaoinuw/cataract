# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:53:52 2022

@author: Yue
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


DATA_PATH = "../data/"
FIGURE_PATH = "../figure/"
feature_file = "Tables_1_2_data.xlsx"
label_file = "label.xlsx"

df_features = pd.read_excel(DATA_PATH + feature_file)
df_label = pd.read_excel(DATA_PATH + label_file)

# df_features = df_features.drop("EPP/LT", axis=1)

X = df_features.values
y = df_label.values
#%%
# replicate Edu's work
y_pred = 8.497 * df_features["EPP/LT"] + 0.25 * df_features["ACD_pre (mm)"]
mae = abs(y_pred - df_label["LP"]).mean()
me = (y_pred - df_label["LP"]).mean()

#%%
pca = PCA(n_components=2)
pca.fit(df_features)

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

ax.xaxis.set_ticks_position('bottom')

# Set ticks on both sides of axes on
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
# Rotate and align bottom ticklabels
plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="center", rotation_mode="anchor")
ax.set_title("Eye Features Correlation")
ax.tick_params(axis='both', labelsize=5)
#fig.tight_layout()
#plt.savefig(FIGURE_PATH+'eye_features_correlation.png', dpi=300, bbox_inches="tight")
plt.show()

