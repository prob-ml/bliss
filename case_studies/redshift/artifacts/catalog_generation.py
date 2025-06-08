import logging
from pathlib import Path

import GCRCatalogs
import numpy as np
import pandas as pd
import tables_io  # pylint: disable=import-error
import yaml
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from GCRCatalogs.helpers.tract_catalogs import tract_filter
from hydra import compose, initialize
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

logging.basicConfig(level=logging.INFO)

#################################
# %% configs
with initialize(config_path="../", version_base=None):
    notebook_cfg = compose("artifact_creation")
dc2_cached_dir = Path(notebook_cfg.paths["dc2_cached"])
dc2_raw_dir = Path(notebook_cfg.paths["dc2"])
cosmodc2_dir = Path(notebook_cfg.paths["cosmodc2"])
plot_dir = Path(notebook_cfg.paths["plots"])

GCRCatalogs.set_root_dir(dc2_raw_dir)
truth_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_truth")
match_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")

tracts = [3828, 3829]


#################################
# get data

# %% truth cat
bands_truth = [f"mag_{band}" for band in "ugrizy"]
truth_df = pd.DataFrame(
    truth_cat.get_quantities(
        bands_truth
        + ["id", "ra", "dec", "redshift", "tract", "patch", "cosmodc2_id", "truth_type"],
        native_filters=[tract_filter(tracts)],
    )
)
logging.info(f"Got {len(truth_df)} objects from tracts {tracts}")

# %% match cat
bands_model = [f"mag_{band}_cModel" for band in "ugrizy"]
errs_model = [f"magerr_{band}_cModel" for band in "ugrizy"]
psf_variables = ["IxxPSF_pixel_", "IyyPSF_pixel_", "IxyPSF_pixel_", "psf_fwhm_"]
psf_columns = [var + band for band in "ugrizy" for var in psf_variables]
match_data = match_cat.get_quantities(
    bands_model
    + errs_model
    + psf_columns
    + ["id_truth", "is_good_match", "match_objectId", "blendedness"],
    native_filters=[tract_filter(tracts)],
)
match_df = pd.DataFrame(match_data)

# report number of matched objects
logging.info(f"Got {len(match_df)} matched objects from tracts {tracts}")

# if is_good_match==False for any, log a warning, drop and indicate percentage dropped
if match_df["is_good_match"].sum() != len(match_df):
    percent_dropped = (len(match_df) - match_df["is_good_match"].sum()) / len(match_df) * 100
    logging.warning(f"Dropping {percent_dropped:.2f}% of rows due to is_good_match==False")
    match_df = match_df[match_df["is_good_match"]]

# if there are any NA rows, count how many and drop them
if match_df.isna().sum().sum() > 0:
    percent_dropped = (match_df.isna().sum().sum() / len(match_df)) * 100
    logging.warning(f"Dropping {percent_dropped:.2f}% of rows due to NaNs in match_df")
    logging.warning(f"Example bad row: {match_df[match_df.isna().any(axis=1)].iloc[0]}")
    match_df = match_df.dropna()

# identify id_truth values that appear more than once
duplicate_ids = match_df["id_truth"].value_counts()
duplicate_ids = duplicate_ids[duplicate_ids > 1].index

# # if there are any duplicates, log a warning
if len(duplicate_ids) > 0:
    percent_duplicates = (len(duplicate_ids) / len(match_df)) * 100
    logging.warning(
        f"Dropping {len(duplicate_ids)} duplicate id_truth values in match_df, "
        f"which is {percent_duplicates:.2f}% of the total"
    )
    match_df = match_df[~match_df["id_truth"].isin(duplicate_ids)]

# report number of matched objects
logging.info(f"After filtering {len(match_df)} matched objects from tracts {tracts} remain")

# %% merge
# Check that each truth_id appears exactly once in truth_df
truth_id_counts = truth_df["id"].value_counts()
if not all(truth_id_counts == 1):
    raise ValueError(
        "Some truth IDs appear multiple times in truth_df. IDs appearing multiple times: "
        f"{str(truth_id_counts[truth_id_counts > 1])}"
    )

# Also verify all match_df truth IDs exist in truth_df
missing_ids = set(match_df["id_truth"]) - set(truth_df["id"])
if missing_ids:
    raise ValueError(f"{len(missing_ids)} truth IDs in match_df are missing from truth_df")

# Merge match_df with truth_df to get redshifts
# Use id_truth from match_df to match with id from truth_df
df = match_df.merge(truth_df, left_on="id_truth", right_on="id", how="left")

# drop unnecessary columns
df = df.drop(columns=["id_truth"])

# rename redshift to redshifts for some reason
df.rename(columns={"redshift": "redshifts"}, inplace=True)

# add some empty columns for the shear
df["shear_1"] = np.nan
df["shear_2"] = np.nan
df["ellipticity_1_true"] = np.nan
df["ellipticity_2_true"] = np.nan
df["cosmodc2_mask"] = False

# rename mag_{} to mag_{}_truth
rename_dict = {f"mag_{b}": f"mag_{b}_truth" for b in "ugrizy" if f"mag_{b}" in df.columns}
df.rename(columns=rename_dict, inplace=True)

# %% Make sure there are no "object" dtypes

# Ensure there are no "object" dtypes in the dataframe
for column in df.select_dtypes(include=["object"]).columns:
    logging.warning(f"Column {column} has dtype 'object'. Converting to string.")
    df[column] = df[column].astype("S")

# Ensure there are no "object" or "string" dtypes in the dataframe
for column in df.columns:
    if df[column].dtype == "object":
        raise ValueError(f"Column {column} has dtype 'object' despite attempt to coerce.")
    if pd.api.types.is_string_dtype(df[column]):
        logging.info(f"Column {column} has 'string' dtype {str(df[column].dtype)}")


##########################
# %% save final product
df.to_parquet(dc2_cached_dir / "merged_catalog.parquet")

#################################
# %% get a little image data a la LSST Science Pipelines (DM Stack) v19.0.0
# (this is just for sanity checking)

relevant_tract = 3828
relevant_patch = "2,3"

band = "r"
deepcoadd_base = dc2_raw_dir / "run2.2i-dr6-v4" / "coadd-t3828-t3829" / "deepCoadd-results"

rawfn = (
    deepcoadd_base
    / band
    / str(relevant_tract)
    / relevant_patch
    / f"calexp-{band}-{relevant_tract}-{relevant_patch}.fits"
)
with fits.open(rawfn) as hdul:
    # Extract the image data
    # image is stored in HDU 1
    image_data = hdul[1].data
    # Extract the WCS information
    wcs = WCS(hdul[1].header)
    # Extract the header information

####################
# %% sanity check against images

# Get the RA/Dec of pixel (100, 100)
pixel_coords = [100, 100]
ra_dec_BR = wcs.pixel_to_world(pixel_coords[0], pixel_coords[1])

# Get the RA/Dec of pixel (0, 0)
pixel_coords = [0, 0]
ra_dec_TL = wcs.pixel_to_world(pixel_coords[0], pixel_coords[1])

# identify values of df where the ra/dec is between these two points
# Define the region of interest (ROI) using the RA/Dec bounds
ra_min, ra_max = min(ra_dec_TL.ra.deg, ra_dec_BR.ra.deg), max(ra_dec_TL.ra.deg, ra_dec_BR.ra.deg)
dec_min, dec_max = min(ra_dec_TL.dec.deg, ra_dec_BR.dec.deg), max(
    ra_dec_TL.dec.deg, ra_dec_BR.dec.deg
)

# Filter the dataframe for objects within the ROI
roi_df = df[(df["ra"].between(ra_min, ra_max)) & (df["dec"].between(dec_min, dec_max))]

# convert the roi_df to pixel coords
pix_x, pix_y = wcs.world_to_pixel(
    SkyCoord(ra=roi_df["ra"].values * units.deg, dec=roi_df["dec"].values * units.deg)
)

logging.info(f"Found {len(roi_df)} objects in the region of interest")

# plot image
plt.pcolormesh(image_data[:100, :100], norm=LogNorm())
plt.plot(pix_x, pix_y, "ro", alpha=0.5)
fn = (
    f"catalog_generation_sanity_check_image_{band}_"
    f"{relevant_tract}_{relevant_patch}_sanity_check.png"
)
plt.savefig(plot_dir / fn)
# %%


#################################
# %% get train/validation/test split
with open(dc2_cached_dir / "splits.yaml", "r") as f:  # pylint: disable=unspecified-encoding
    splits = yaml.safe_load(f)

trainsets = [["_".join(x) for x in y] for y in splits["train"]]
testsets = [["_".join(x) for x in y] for y in splits["test"]]

# Ensure train and test splits do not overlap
for i, (a, b) in enumerate(zip(trainsets, testsets)):
    if len(a) != len(set(a)):
        raise ValueError(f"Train split {i} has duplicates: {a}")
    if len(b) != len(set(b)):
        raise ValueError(f"Test split {i} has duplicates: {b}")
    overlap = set(a) & set(b)
    if overlap:
        raise ValueError(f"Train and test splits overlap for index {i}: {overlap}")


# measure overlap between the test splits
for i, a in enumerate(testsets):
    for j, b in enumerate(testsets):
        if i <= j:
            continue
        overlap = set(a) & set(b)
        if overlap:
            logging.info(f"Testsets {i} and {j} overlap by {len(overlap)}")

# %% establish relevant columns, create appropriate dataframe
columns_to_include = [f"mag_{x}_cModel" for x in "ugrizy"]
columns_to_include += [f"magerr_{x}_cModel" for x in "ugrizy"]
columns_to_include += ["ra", "dec", "tract", "patch", "redshifts"]
columns_to_include += [f"mag_{x}_truth" for x in "ugrizy"]
df_for_rail = df[columns_to_include].copy()
df_for_rail.rename(columns=lambda x: x.replace("cModel", "lsst"), inplace=True)
df_for_rail.rename(columns=lambda x: x.replace("magerr", "mag_err"), inplace=True)
df_for_rail.rename(columns=lambda x: x.replace("redshifts", "redshift"), inplace=True)

# drop things that aren't galaxies
galaxies = df["truth_type"] == 1
df_for_rail = df_for_rail[galaxies]


#################################
# %% read ttsplits for the purposes of making rail-friendly train/test splits

# create tract_patch series
tract_patch = df_for_rail["tract"].astype(str) + "_" + df_for_rail["patch"].astype(str)

for i, (train, test) in enumerate(zip(trainsets, testsets)):
    train_df = df_for_rail[tract_patch.isin(train)]
    test_df = df_for_rail[tract_patch.isin(test)]

    assert len(train_df) > 900_000
    assert len(test_df) > 150_000

    logging.info(f"Processing split {i}")
    logging.info(f"Training set: {len(train_df)} samples")
    logging.info(f"Test set: {len(test_df)} samples")

    tables_io.write(train_df, dc2_cached_dir / f"rail_train_split_{i}.hdf5", "hdf5")
    tables_io.write(test_df, dc2_cached_dir / f"rail_test_split_{i}.hdf5", "hdf5")

#################################
# %% a tiny test set for testing

tables_io.write(test_df.iloc[:500], dc2_cached_dir / "rail_tiny_test.hdf5", "hdf5")

# %%
