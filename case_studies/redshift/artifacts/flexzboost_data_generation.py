import logging
import pickle
from pathlib import Path

import GCRCatalogs
import pandas as pd
import tables_io  # pylint: disable=import-error
from GCRCatalogs.helpers.tract_catalogs import tract_filter
from hydra import compose, initialize

logging.basicConfig(level=logging.INFO)

#################################
# %% configs
with initialize(config_path="../", version_base=None):
    notebook_cfg = compose("redshift")
rail_dir = Path(notebook_cfg.paths["processed_data_dir_rail"])
bliss_dir = Path(notebook_cfg.paths["processed_data_dir_bliss"])
out_model_fn = rail_dir / "flexzboost_model_results.pkl"

#################################
# %% get train/validation/test split
with open(bliss_dir / "train_splits.pkl", "rb") as f:
    train_splits = ["_".join(x) for x in pickle.load(f)]

with open(bliss_dir / "test_splits.pkl", "rb") as f:
    test_splits = ["_".join(x) for x in pickle.load(f)]

#################################
# %% Set GCR root directory from config
GCRCatalogs.set_root_dir(notebook_cfg.rail.pipeline.lsst_root_dir)
truth_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_truth")
obj_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object")
match_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")

#################################
# %% get catalog data
tracts = [3828, 3829]

bands = ["mag_u_truth", "mag_g_truth", "mag_r_truth", "mag_i_truth", "mag_z_truth", "mag_y_truth"]
bands_model = [
    "mag_u_cModel",
    "mag_g_cModel",
    "mag_r_cModel",
    "mag_i_cModel",
    "mag_z_cModel",
    "mag_y_cModel",
]
errs_model = [
    "magerr_u_cModel",
    "magerr_g_cModel",
    "magerr_r_cModel",
    "magerr_i_cModel",
    "magerr_z_cModel",
    "magerr_y_cModel",
]

# truth data
truth_data = truth_cat.get_quantities(
    ["redshift", "id", "truth_type"],
    native_filters=[tract_filter(tracts)],
)
truth_df = pd.DataFrame(truth_data)
logging.info(f"Got {len(truth_df)} ground truth objects from tracts {tracts}")

# object-truth match data
match_data = match_cat.get_quantities(
    bands
    + bands_model
    + errs_model
    + ["match_objectId", "id_truth", "patch", "x", "y", "tract", "ra", "dec"],
    native_filters=[tract_filter(tracts)],
    filters=["is_good_match==True"],
)
match_df = pd.DataFrame(match_data)

# drop rows with NaNs
match_df = match_df.dropna()

# report number of matched objects
logging.info(f"Got {len(match_df)} matched objects from tracts {tracts}")

#################################
# %% postprocessing the catalogs a bit

# Check that each truth_id appears exactly once in truth_df
truth_id_counts = truth_df["id"].value_counts()
if not all(truth_id_counts == 1):
    raise ValueError(
        "Some truth IDs appear multiple times in truth_df. IDs appearing multiple times: "
        + str(truth_id_counts[truth_id_counts > 1])
    )

# Also verify all match_df truth IDs exist in truth_df
missing_ids = set(match_df["id_truth"]) - set(truth_df["id"])
if missing_ids:
    raise ValueError(f"{len(missing_ids)} truth IDs in match_df are missing from truth_df")

# Merge match_df with truth_df to get redshifts
# Use id_truth from match_df to match with id from truth_df
df = match_df.merge(
    truth_df[["id", "redshift", "truth_type"]], left_on="id_truth", right_on="id", how="left"
)

# drop all objects that are not type 1 (galaxies)
df = df[df["truth_type"] == 1]

# create jointed tract,patch column
df["tract_patch"] = pd.Categorical(df["tract"].astype(str) + "_" + df["patch"].astype(str))

# patch is bytestring, not unicode
df["patch"] = df.patch.astype("S4")

# report number of sources
logging.info(f"Got {len(df)} total sources")


#################################
# %% make ttsplit

# create the datasets
columns_to_include = [
    "mag_u_cModel",
    "mag_g_cModel",
    "mag_r_cModel",
    "mag_i_cModel",
    "mag_z_cModel",
    "mag_y_cModel",
    "redshift",
    "x",
    "y",
    "ra",
    "dec",
    "tract",
    "patch",
    "magerr_u_cModel",
    "magerr_g_cModel",
    "magerr_r_cModel",
    "magerr_i_cModel",
    "magerr_z_cModel",
    "magerr_y_cModel",
]

train_df = df[df["tract_patch"].isin(train_splits)][columns_to_include]
test_df = df[df["tract_patch"].isin(test_splits)][columns_to_include]

# adopt naming convention from rail
# mag_<band>_cModel --> mag_<band>_lsst"
train_df.rename(columns=lambda x: x.replace("cModel", "lsst"), inplace=True)
test_df.rename(columns=lambda x: x.replace("cModel", "lsst"), inplace=True)

# magerr -> mag_err
train_df.rename(columns=lambda x: x.replace("magerr", "mag_err"), inplace=True)
test_df.rename(columns=lambda x: x.replace("magerr", "mag_err"), inplace=True)

logging.info(f"Training set: {len(train_df)} samples")
logging.info(f"Test set: {len(test_df)} samples")

#################################
# %% save to format that can be read by rail
results_dir = rail_dir
tables_io.write(train_df, rail_dir / "train_df.hdf5", "hdf5")
tables_io.write(test_df, rail_dir / "test_df.hdf5", "hdf5")
tables_io.write(test_df.iloc[:100], rail_dir / "test_df_tiny.hdf5", "hdf5")
