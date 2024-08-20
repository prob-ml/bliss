import os
import pickle as pkl

import GCRCatalogs
import healpy as hp
import numpy as np
import pandas as pd
from GCRCatalogs import GCRQuery

GCRCatalogs.set_root_dir("/data/scratch/dc2_nfs/")

file_name = "dc2_lensing_catalog_alt.pkl"
file_path = os.path.join("/data", "scratch", "dc2local", file_name)
file_already_populated = os.path.isfile(file_path)

if file_already_populated:
    raise FileExistsError(f"{file_path} already exists.")


print("Loading truth...\n")  # noqa: WPS421

truth_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_truth")

truth_df = truth_cat.get_quantities(
    quantities=[
        "cosmodc2_id",
        "id",
        "match_objectId",
        "truth_type",
        "ra",
        "dec",
        "redshift",
        "flux_u",
        "flux_g",
        "flux_r",
        "flux_i",
        "flux_z",
        "flux_y",
        "mag_u",
        "mag_g",
        "mag_r",
        "mag_i",
        "mag_z",
        "mag_y",
    ]
)
truth_df = pd.DataFrame(truth_df)

truth_df = truth_df[truth_df["truth_type"] == 1]

truth_df = truth_df[truth_df["flux_r"] >= 50]

max_ra = np.nanmax(truth_df["ra"])
min_ra = np.nanmin(truth_df["ra"])
max_dec = np.nanmax(truth_df["dec"])
min_dec = np.nanmin(truth_df["dec"])
ra_dec_filters = [f"ra >= {min_ra}", f"ra <= {max_ra}", f"dec >= {min_dec}", f"dec <= {max_dec}"]

vertices = hp.ang2vec(
    np.array([min_ra, max_ra, max_ra, min_ra]),
    np.array([min_dec, min_dec, max_dec, max_dec]),
    lonlat=True,
)
ipix = hp.query_polygon(32, vertices, inclusive=True)
healpix_filter = GCRQuery((lambda h: np.isin(h, ipix, assume_unique=True), "healpix_pixel"))


print("Loading object-with-truth-match...\n")  # noqa: WPS421

object_truth_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")

object_truth_df = object_truth_cat.get_quantities(
    quantities=[
        "cosmodc2_id_truth",
        "Ixx_pixel",
        "Iyy_pixel",
        "Ixy_pixel",
        "IxxPSF_pixel_u",
        "IxxPSF_pixel_g",
        "IxxPSF_pixel_r",
        "IxxPSF_pixel_i",
        "IxxPSF_pixel_z",
        "IxxPSF_pixel_y",
        "IyyPSF_pixel_u",
        "IyyPSF_pixel_g",
        "IyyPSF_pixel_r",
        "IyyPSF_pixel_i",
        "IyyPSF_pixel_z",
        "IyyPSF_pixel_y",
        "IxyPSF_pixel_u",
        "IxyPSF_pixel_g",
        "IxyPSF_pixel_r",
        "IxyPSF_pixel_i",
        "IxyPSF_pixel_z",
        "IxyPSF_pixel_y",
        "psf_fwhm_u",
        "psf_fwhm_g",
        "psf_fwhm_r",
        "psf_fwhm_i",
        "psf_fwhm_z",
        "psf_fwhm_y",
    ]
)
object_truth_df = pd.DataFrame(object_truth_df)


print("Loading CosmoDC2...\n")  # noqa: WPS421

config_overwrite = {"catalog_root_dir": "/data/scratch/dc2_nfs/cosmoDC2"}

cosmo_cat = GCRCatalogs.load_catalog("desc_cosmodc2", config_overwrite)

cosmo_df = cosmo_cat.get_quantities(
    quantities=[
        "galaxy_id",
        "ra",
        "dec",
        "ellipticity_1_true",
        "ellipticity_2_true",
        "shear_1",
        "shear_2",
        "convergence",
    ],
    filters=ra_dec_filters,
    native_filters=healpix_filter,
)
cosmo_df = pd.DataFrame(cosmo_df)


print("Merging truth with object-with-truth-match...\n")  # noqa: WPS421

merge_df1 = truth_df.merge(
    object_truth_df, left_on="cosmodc2_id", right_on="cosmodc2_id_truth", how="left"
)

merge_df1.drop_duplicates(subset=["cosmodc2_id"], inplace=True)

merge_df1.drop(columns=["cosmodc2_id_truth"], inplace=True)


print("Merging with CosmoDC2...\n")  # noqa: WPS421

merge_df2 = merge_df1.merge(cosmo_df, left_on="cosmodc2_id", right_on="galaxy_id", how="left")

merge_df2 = merge_df2[~merge_df2["galaxy_id"].isna()]

merge_df2.drop(columns=["ra_y", "dec_y"], inplace=True)

merge_df2.rename(columns={"ra_x": "ra", "dec_x": "dec"}, inplace=True)


print("Saving...\n")  # noqa: WPS421

with open(file_path, "wb") as f:
    pkl.dump(merge_df2, f)

print(f"Catalog has been saved at {file_path}")  # noqa: WPS421
