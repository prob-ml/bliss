# %% imports
import logging
import pickle
from pathlib import Path

import flexzboost_fast
import numpy as np
import pandas as pd
import rail  # pylint: disable=import-error
from hydra import compose, initialize
from matplotlib import pyplot as plt
from rail.core.stage import RailStage  # pylint: disable=import-error
from rail.estimation.algos.flexzboost import FlexZBoostEstimator  # pylint: disable=import-error

logging.basicConfig(level=logging.INFO)
RailStage.data_store.__class__.allow_overwrite = True

# %% configs
with initialize(config_path="../", version_base=None):
    cfg = compose("redshift")
rail_dir = Path(cfg.paths["processed_data_dir_rail"])
out_model_fn = rail_dir / "flexzboost_model_results.pkl"

# %% Load the model from the pickle file
with open(out_model_fn, "rb") as model_file:
    model = pickle.load(model_file)

# %% make a stage for running the model
outfn = rail_dir / "flexzboost_predictions.hdf5"
pzflex_qp_flexzboost = FlexZBoostEstimator.make_stage(
    name="fzboost_flexzboost",
    hdf5_groupname=None,
    model=model,
    output=outfn,
    qp_representation="flexzboost",
)

# %% load tiny testing data
test_data = RailStage.data_store.read_file(
    "training_data", rail.core.data.Hdf5Handle, rail_dir / "test_df.hdf5"
)

# %% make pdf predictions
predictions = pzflex_qp_flexzboost.estimate(test_data)

# %% grid them out
xs, pdfs = flexzboost_fast.compute_gridded_pdfs(predictions())

# %% compute ground truth for each sample
y_test = test_data.read()["redshift"]
y_mags = test_data.read()["mag_i_lsst"]  # maybe should use ground truth instead?

# %% loss reducing predictions
loss_types = {
    "catastrophic": flexzboost_fast.CatastrophicOutlierLoss(),
    "L2": flexzboost_fast.MeanSquaredLoss(),
    "L1": flexzboost_fast.MeanAbsoluteLoss(),
    "one_plus": flexzboost_fast.OnePlusOutlierLoss(),
}

# %% get predictions, various ways
prediction_types = {
    "L2": flexzboost_fast.estimate_loss_minimizers(xs, pdfs, loss_types["L2"]),
    "L1": flexzboost_fast.estimate_loss_minimizers(xs, pdfs, loss_types["L1"]),
    "one_plus": flexzboost_fast.estimate_loss_minimizers(xs, pdfs, loss_types["one_plus"]),
    "catastrophic": flexzboost_fast.estimate_loss_minimizers(xs, pdfs, loss_types["catastrophic"]),
}


# %% divide ground truth into bins by mag and redshift
def process_cutoff_names(cutoffs):
    cutoff_names = ["<" + str(cutoffs[0])]
    for lower, upper in zip(cutoffs[:-1], cutoffs[1:]):
        cutoff_names.append(f"{lower}-{upper}")
    cutoff_names.append(">" + str(cutoffs[-1]))
    return cutoff_names


mag_bin_cutoffs = cfg.visualization.mag_bin_cutoffs
y_test_mag_index = np.digitize(y_mags, mag_bin_cutoffs)
mag_bin_cutoff_names = process_cutoff_names(mag_bin_cutoffs)

redshift_bin_cutoffs = cfg.visualization.redshift_bin_cutoffs
y_test_redshift_index = np.digitize(y_test, redshift_bin_cutoffs)
redshift_bin_cutoff_names = process_cutoff_names(redshift_bin_cutoffs)


def compute_loss_by_bin(bin_cutoffs, bin_names, bin_indices, binning_name):
    results = []
    for loss_name, loss in loss_types.items():
        for prediction_name, prediction in prediction_types.items():
            for bin_index in range(0, len(bin_cutoffs) + 1):
                salient = bin_indices == bin_index
                loss_values = loss(y_test[salient], prediction[salient])
                logging.info(
                    f"Computed {loss_name} loss for {prediction_name} in {binning_name} bin "
                    f"{bin_names[bin_index]}: {np.mean(loss_values)}"
                )
                results.append(
                    {
                        "loss_type": loss_name,
                        "prediction_type": prediction_name,
                        "loss": np.mean(loss_values),
                        "bin": bin_names[bin_index],
                        "bin_index": bin_index,
                        "binning_name": binning_name,
                    }
                )
    return pd.DataFrame(results)


results_by_mag = compute_loss_by_bin(mag_bin_cutoffs, mag_bin_cutoff_names, y_test_mag_index, "mag")
results_by_redshift = compute_loss_by_bin(
    redshift_bin_cutoffs, redshift_bin_cutoff_names, y_test_redshift_index, "redshift"
)

# %% save the results
results_by_mag.to_csv(rail_dir / "results_by_mag.csv")
results_by_redshift.to_csv(rail_dir / "results_by_redshift.csv")

# %% focusing on the "diagonal" entries of the loss table
results_by_mag_diagonal = results_by_mag[
    results_by_mag["loss_type"] == results_by_mag["prediction_type"]
]
results_by_redshift_diagonal = results_by_redshift[
    results_by_redshift["loss_type"] == results_by_redshift["prediction_type"]
]

# %%
# %% plot redshift_bin against loss for each loss type in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, loss_name in enumerate(loss_types.keys()):
    subset = results_by_redshift_diagonal[results_by_redshift_diagonal["loss_type"] == loss_name]
    axs[i].plot(subset["bin"], subset["loss"], label=loss_name)
    axs[i].set_xlabel("Redshift Bin")
    axs[i].set_ylabel("Loss")
    axs[i].set_title(f"Loss vs Redshift Bin for {loss_name}")
    axs[i].tick_params(axis="x", rotation=45)
    axs[i].set_ylim(0, None)
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(rail_dir / "loss_vs_redshift_bin.png")
# %%
