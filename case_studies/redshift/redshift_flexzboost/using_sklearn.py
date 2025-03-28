# %% imports
from pathlib import Path

import numpy as np
import pandas as pd
import tables_io  # pylint: disable=import-error
from hydra import compose, initialize
from sklearn.ensemble import HistGradientBoostingRegressor

# %% configs
with initialize(config_path="../", version_base=None):
    notebook_cfg = compose("redshift_flexzboost")
rail_dir = Path(notebook_cfg.paths["processed_data_dir_rail"])
out_model_fn = rail_dir / "flexzboost_model_results.pkl"

# %% load training data
train_df = pd.DataFrame(tables_io.read(rail_dir / "train_df.hdf5"))
test_df = pd.DataFrame(tables_io.read(rail_dir / "test_df.hdf5"))

# get bands of interest
bands = ["u", "g", "r", "i", "z"]
bands_model = [f"mag_{band}_lsst" for band in bands]

# %% train sklearn gradienthistogramregressor
# to predict redshift rom mag_{band}_lsst

model = HistGradientBoostingRegressor()
model.fit(train_df[bands_model], train_df["redshift"])

# %% make predictions on test
preds = model.predict(test_df[bands_model])


# %% divide ground truth into bins by mag and redshift
def process_cutoff_names(cutoffs):
    cutoff_names = ["<" + str(cutoffs[0])]
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        cutoff_names.append(f"{lower}-{upper}")
    cutoff_names.append(">" + str(cutoffs[-1]))
    return cutoff_names


y_test = test_df["redshift"]
y_mags = test_df["mag_i_lsst"]  # maybe should use ground truth instead?

mag_bin_cutoffs = notebook_cfg.visualization.mag_bin_cutoffs
y_test_mag_index = np.digitize(y_mags, mag_bin_cutoffs)
mag_bin_cutoff_names = process_cutoff_names(mag_bin_cutoffs)

redshift_bin_cutoffs = notebook_cfg.visualization.redshift_bin_cutoffs
y_test_redshift_index = np.digitize(y_test, redshift_bin_cutoffs)
redshift_bin_cutoff_names = process_cutoff_names(redshift_bin_cutoffs)

# %%
# collate mean l2 loss for each mag bin
mag_bin_loss = np.zeros(len(mag_bin_cutoffs) + 1)
for i in range(len(mag_bin_cutoffs) + 1):
    mask = y_test_mag_index == i
    mag_bin_loss[i] = np.mean((y_test[mask] - preds[mask]) ** 2)

# and redhisft bin
redshift_bin_loss = np.zeros(len(redshift_bin_cutoffs) + 1)
for i in range(len(redshift_bin_cutoffs) + 1):
    mask = y_test_redshift_index == i
    redshift_bin_loss[i] = np.mean((y_test[mask] - preds[mask]) ** 2)

# %% save results

mag_results = pd.DataFrame(
    {
        "loss_type": ["L2" for _ in mag_bin_loss],
        "loss": mag_bin_loss,
        "bin": mag_bin_cutoff_names,
        "bin_index": list(range(len(mag_bin_loss))),
        "binning_name": ["mag" for _ in mag_bin_loss],
    }
)

mag_results.to_csv(rail_dir / "sklearn_results_by_mag.csv")

redshift_results = pd.DataFrame(
    {
        "loss_type": ["L2" for _ in redshift_bin_loss],
        "loss": redshift_bin_loss,
        "bin": redshift_bin_cutoff_names,
        "bin_index": list(range(len(redshift_bin_loss))),
        "binning_name": ["redshift" for _ in redshift_bin_loss],
    }
)

redshift_results.to_csv(rail_dir / "sklearn_results_by_redshift.csv")
# %%
