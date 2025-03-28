# %% imports
import logging
import pickle
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from matplotlib import pyplot as plt
from rail.core.stage import RailStage  # pylint: disable=import-error

logging.basicConfig(level=logging.INFO)
RailStage.data_store.__class__.allow_overwrite = True

# %% configs
with initialize(config_path="../", version_base=None):
    cfg = compose("redshift_flexzboost")
rail_dir = Path(cfg.paths["processed_data_dir_rail"])
out_model_fn = rail_dir / "flexzboost_model_results.pkl"

# %% load csvs
results_by_mag = pd.read_csv(rail_dir / "results_by_mag.csv", index_col=0)
results_by_redshift = pd.read_csv(rail_dir / "results_by_redshift.csv", index_col=0)

# %% get unique loss types (categories of loss_type column)
loss_types = results_by_redshift["loss_type"].unique()


# %% focusing on the "diagonal" entries of the loss table
results_by_mag_diagonal = results_by_mag[
    results_by_mag["loss_type"] == results_by_mag["prediction_type"]
]
results_by_redshift_diagonal = results_by_redshift[
    results_by_redshift["loss_type"] == results_by_redshift["prediction_type"]
]

# %% plot redshift_bin against loss for each loss type in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, loss_name in enumerate(loss_types):
    subset = results_by_redshift_diagonal[results_by_redshift_diagonal["loss_type"] == loss_name]
    axs[i].plot(subset["bin"], subset["loss"])
    axs[i].set_xlabel("Redshift Bin")
    axs[i].set_ylabel("Loss")
    axs[i].set_title(f"Loss vs Redshift Bin for {loss_name}")
    axs[i].tick_params(axis="x", rotation=45)
    axs[i].set_ylim(0, None)
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(rail_dir / "loss_vs_redshift_bin.png")

# %% plot mag_bin against loss for each loss type in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, loss_name in enumerate(loss_types):
    subset = results_by_mag_diagonal[results_by_mag_diagonal["loss_type"] == loss_name]
    axs[i].plot(subset["bin"], subset["loss"])
    axs[i].set_xlabel("Magnitude Bin")
    axs[i].set_ylabel("Loss")
    axs[i].set_title(f"Loss vs Magnitude Bin for {loss_name}")
    axs[i].tick_params(axis="x", rotation=45)
    axs[i].set_ylim(0, None)
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(rail_dir / "loss_vs_mag_bin.png")

# %% compare against bliss (by magnitude)
naming_convention = {
    "catastrophic": "redshifts/outlier_fraction_cata",
    "L2": "redshifts/mse",
    "one_plus": "redshifts/outlier_fraction",
}
data_dir = Path(cfg.paths["data_dir"])
with open(data_dir / "plots" / "cts_mode_metrics_45.pkl", "rb") as f:
    bliss_cts_results = pickle.load(f)


# %% plot mag_bin against loss for each loss type in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, loss_name in enumerate(loss_types):
    subset = results_by_mag_diagonal[results_by_mag_diagonal["loss_type"] == loss_name]
    axs[i].plot(subset["bin_index"], subset["loss"], label="FlexZBoost")

    axs[i].set_xticks(subset["bin_index"], subset["bin"])

    # add bliss results
    bliss_values = []
    loss_name_modified = naming_convention.get(loss_name)
    if loss_name_modified is not None:
        for j in range(6):
            key = f"{loss_name_modified}_bin_{j}"
            bliss_values.append(bliss_cts_results[key])
        axs[i].plot(range(6), bliss_values, label="Bliss (normal)")

    axs[i].set_xlabel("Magnitude Bin")
    axs[i].set_ylabel("Loss")
    axs[i].set_title(f"Loss vs Magnitude Bin for {loss_name}")
    axs[i].tick_params(axis="x", rotation=45)
    axs[i].set_ylim(0, None)
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.savefig(rail_dir / "loss_vs_mag_bin_comparison.png")

# %%
