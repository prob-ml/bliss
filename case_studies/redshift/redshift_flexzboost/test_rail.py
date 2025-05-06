import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rail  # pylint: disable=import-error
import redshift_losses
from hydra import compose, initialize
from rail.core.stage import RailStage  # pylint: disable=import-error
from rail.estimation.algos.flexzboost import FlexZBoostEstimator  # pylint: disable=import-error

logging.basicConfig(level=logging.INFO)
RailStage.data_store.__class__.allow_overwrite = True

# %% configs
with initialize(config_path="../", version_base=None):
    cfg = compose("artifact_creation")
rail_dir = Path(cfg.paths["rail_checkpoints"])
cached_dc2 = Path(cfg.paths["dc2_cached"])
plots_dir = Path(cfg.paths["plots"])

redshift_bins = np.r_[0:3:1800j]

# %% get all filenames like rail_dir / train_df_{i}
# listing all files in the directory
train_df_files = list(cached_dc2.glob("rail_train_split_*.hdf5"))
# sorting the files by their index
train_df_files.sort(key=lambda x: int(x.stem.split("_")[-1]))

rng = np.random.default_rng(42)

# %% loop over test/train splits
for i, train_df_file in enumerate(train_df_files):
    logging.info(f"Running test set {i}")

    # %% Load the model from the pickle file
    with open(rail_dir / f"flexzboost_model_results_split_{i}.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # %% make a stage for running the model
    outfn = rail_dir / f"flexzboost_pdfs_split_{i}.hdf5"
    pzflex_qp_flexzboost = FlexZBoostEstimator.make_stage(
        name="fzboost_flexzboost",
        hdf5_groupname=None,
        model=model,
        output=outfn,
        qp_representation="flexzboost",
    )

    # %% load testing data
    test_data = RailStage.data_store.read_file(
        "testing_data", rail.core.data.Hdf5Handle, cached_dc2 / f"rail_test_split_{i}.hdf5"
    )
    # test_data = RailStage.data_store.read_file(
    #     "testing_data", rail.core.data.Hdf5Handle, cached_dc2 / f"rail_tiny_test.hdf5"
    # )

    # %% get ground truth and binning information
    y_test = test_data.read()["redshift"]
    y_mags = test_data.read()["mag_r_truth"]

    # %% make pdf predictions
    start_time = time.time()
    predictions = pzflex_qp_flexzboost.estimate(test_data)
    pdfs = predictions.data.pdf(redshift_bins)
    end_time = time.time()
    logging.info(f"Time taken to compute PDFs: {end_time - start_time} seconds")

    # %% make the pdfs into pmfs
    pdfs = pdfs / np.sum(pdfs, axis=1, keepdims=True)

    # %% sample a sample from each
    samples1 = np.array([rng.choice(redshift_bins, p=pdf) for pdf in pdfs])
    samples2 = np.array([rng.choice(redshift_bins, p=pdf) for pdf in pdfs])

    # %% get mode estimates
    mode_indices = np.argmax(pdfs, axis=1)
    modes = redshift_bins[mode_indices]

    # %% compute PIT values
    # (probability integral transform)
    # Compute PIT values
    cdf = np.cumsum(pdfs, axis=1)
    pit_values = np.array(
        [cdf[i, np.searchsorted(redshift_bins, y_test[i]) - 1] for i in range(len(y_test))]
    )

    # %% compute IQR
    # Compute the IQR for each sample
    iqr_values = []
    for row in cdf:
        q1_idx = np.searchsorted(row, 0.25)
        q3_idx = np.searchsorted(row, 0.75)
        q1 = redshift_bins[q1_idx] if q1_idx < len(redshift_bins) else redshift_bins[-1]
        q3 = redshift_bins[q3_idx] if q3_idx < len(redshift_bins) else redshift_bins[-1]
        iqr_values.append(q3 - q1)
    iqr_values = np.array(iqr_values)

    # %% loss reducing predictions
    loss_types = {
        "z_pred_catastrophic": redshift_losses.CatastrophicOutlierLoss(),
        "z_pred_L2": redshift_losses.MeanSquaredLoss(),
        "z_pred_L1": redshift_losses.MeanAbsoluteLoss(),
        "z_pred_outlier": redshift_losses.OnePlusOutlierLoss(),
    }

    # %% get predictions, various ways
    prediction_types = {
        name: redshift_losses.estimate_loss_minimizers(redshift_bins, pdfs, loss)
        for name, loss in loss_types.items()
    }

    # %% save associated parquet file
    outfn = rail_dir / f"flexzboost_predictions_split_{i}.parquet"
    df = pd.DataFrame(
        {
            "z_true": y_test,
            "mag_r_true": y_mags,
            "pit_values": pit_values,
            "iqr_values": iqr_values,
            "z_pred_mode": modes,
            "z_pred_sample1": samples1,
            "z_pred_sample2": samples2,
            **{x: prediction_types[x] for x in prediction_types},
        }
    )
    df.to_parquet(outfn, index=False)
# %%
