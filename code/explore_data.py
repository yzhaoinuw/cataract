# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:53:52 2022

@author: Yue
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor

from neural_network import MLP, Linear

DATA_PATH = "../data/"
FIGURE_PATH = "../figure/"
feature_file = "features_processed_feb.xlsx"

# df_features = pd.read_excel(DATA_PATH + feature_file)
# df_label = pd.read_excel(DATA_PATH + label_file)

# new data
df_features = pd.read_excel(DATA_PATH + feature_file)
#df_features = df_features.dropna()
# select subset of features
use_features = [
    #"AxialLengthmm",
    "RAC",
    "IOLModel_1",
    "IOLModel_2",
    "IOLModel_3",
    #"Sex_1",
    #"Sex_2",
    # Axial measurements, col 17 - 20
    "CT",
    "ACD",
    "LT",
    "VCD",
    # new AL
    "AL",
    # crystalline lens params, set I, col 26-27
    #"MedRALEyes",
    #"MedRPLEyes",
    # crystalline lens params, set II, col 30-31
    #"MedRALEyesDiam2",
    #"MedRPLEyesDiam2",
    # crystalline lens params, set III, col 34-35
    #"RAL3D",
    #"RPL3D",
    # crystalline lens params, set IV, col 38-39
    #"RAL3DDiam2",
    #"RPL3DDiam2",
    # additional features
    #"PupilSize",
    # LP
    "LP",
]
# df_features = df_features[df_features["IOLModel_1"] == 1]
# df_features = df_features.drop("IOLModel_1", axis=1)
df_features = df_features.drop("IOLPowerInsertedD", axis=1)
df_features = df_features[use_features]
df_labels = df_features["LP"].to_frame()
df_features = df_features.drop("LP", axis=1)
df_features = df_features.dropna()
# try some transformation
#df_labels = (df_labels - 4) * 1

X = df_features.iloc[:41, :]
y = df_labels.iloc[:41, :]

#%%
losses = []
coefs = []
mae_losses = []
for i in range(1000):
    if i % 100 == 0:
        print(f"experiment {i}.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    X_test = df_features.iloc[41:, :]
    y_test = df_labels.iloc[41:, :]
    # PCA
    # pca = PCA(n_components=5)
    # pca.fit(df_features.values)
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    
    #kernel = ConstantKernel() * Matern(nu=0.5)
    #model = GaussianProcessRegressor(kernel=kernel)
    model = Ridge()
    #model = Lasso() # much worse than Ridge
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    indices = y_test.index
    coefs.append(model.coef_)
    mse = abs(y_pred-y_test).values.flatten()
    mae_losses.append(mse.mean())
    loss = np.zeros(len(y))
    #loss[y_test.index] = mse
    losses.append(loss)
    
losses = np.array(losses)
mae_losses = np.array(mae_losses)
coefs = np.array(coefs)

#%%
"""
SAVE_LOC = "../results/"
df_mae = pd.DataFrame(losses, columns=list(range(1, len(losses[0])+1)))
df_mae.index = np.arange(1, len(df_mae) + 1)
df_mae.to_excel(SAVE_LOC+"ridge_MAE_RAC+new_AL+IOL_Model+axial_measurements.xlsx", index=False)
#%%

coefs = pd.DataFrame(np.mean(abs(coefs), axis=0), columns=["Coefficients"], index=df_features.columns)

coefs.plot(kind="barh", figsize=(9, 7))
plt.title("Ridge model")
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
#%%

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
