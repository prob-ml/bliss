import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from hydra import compose, initialize
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


# %% configs
with initialize(config_path="../", version_base=None):
    notebook_cfg = compose("artifact_creation")
rail_dir = Path(notebook_cfg.paths["rail_checkpoints"])
cached_dc2 = Path(notebook_cfg.paths["dc2_cached"])

train_df_file = cached_dc2 / "rail_train_split_0.hdf5"
test_df_file = cached_dc2 / "rail_test_split_0.hdf5"

# %% load train_df_file

columns_to_include = [f"mag_{x}_lsst" for x in "ugrizy"]

with h5py.File(train_df_file, "r") as f:
    X = np.array([f[col][:] for col in columns_to_include]).T
    Y = f["redshift"][:]

################################################################
# %% use some of the data to estimate E[Y|X] with histgradientboosting
# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# Initialize and train the model
model = HistGradientBoostingRegressor()
model.fit(X_train, Y_train)

# Evaluate the model on the validation set
score = model.score(X_val, Y_val)
logging.info(f"Validation R^2 score: {score}")

####################################################################
# %% use remaining data to estimate var[Y|f(X)=s] where f(x)=E[Y|X]
# Get the predictions for the validation set
Y_pred = model.predict(X_val)
# Get the residuals
residuals = Y_val - Y_pred
# Model residuals as a function of Y_pred
# Train a new model to estimate the variance of the residuals
model_var = HistGradientBoostingRegressor()
# model_var.fit(X_val, residuals**2)
model_var.fit(Y_pred.reshape(-1, 1), residuals**2)

####################################################################
# %% load test data
with h5py.File(test_df_file, "r") as f:
    X_te = np.array([f[col][:] for col in columns_to_include]).T
    Y_te = f["redshift"][:]
    mag_te = f["mag_r_truth"][:]

####################################################################
# %% predict the mean and variance of the test set
Y_te_pred = model.predict(X_te)
# Y_te_var = model_var.predict(X_te)
Y_te_var = model_var.predict(Y_te_pred.reshape(-1, 1))

####################################################################
# %% compute PIT values

# Compute standard deviations, avoiding negative variance
sigma = np.sqrt(np.clip(Y_te_var, 1e-5, None))

# PIT = CDF of the true value under the predictive normal distribution
pit = norm.cdf((Y_te - Y_te_pred) / sigma)

####################################################################
# %% draw a sample and store in rail_dir / normalboost_predictions_split_0.parquet

# draw one normal sample per test point
samples1 = np.random.normal(loc=Y_te_pred, scale=np.sqrt(np.clip(Y_te_var, 0, None)))
samples2 = np.random.normal(loc=Y_te_pred, scale=np.sqrt(np.clip(Y_te_var, 0, None)))

# assemble dataframe
df = pd.DataFrame(
    {
        "z_pred_L2": Y_te_pred,
        "z_pred_var": Y_te_var,
        "z_pred_sample1": samples1,
        "z_pred_sample2": samples2,
        "z_true": Y_te,
        "mag_r_true": mag_te,
        "pit_values": pit,
    }
)

# write to parquet
output_file = rail_dir / "normalboost_predictions_split_0.parquet"
df.to_parquet(output_file, index=False)
logging.info(f"Saved predictions to {output_file}")

# %%
