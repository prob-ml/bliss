# pylint: disable=R0801
import os
import pickle as pkl

import GCRCatalogs
import healpy as hp
import numpy as np
import pandas as pd
from GCRCatalogs import GCRQuery
from GCRCatalogs.helpers.tract_catalogs import tract_filter

GCRCatalogs.set_root_dir("/data/scratch/dc2_nfs/")

file_name = "dc2_lensing_catalog_objectmatch.pkl"
file_path = os.path.join("/data", "scratch", "dc2local", file_name)
file_already_populated = os.path.isfile(file_path)

if file_already_populated:
    raise FileExistsError(f"{file_path} already exists.")


print("Loading object-with-truth-match...\n")  # noqa: WPS421

object_truth_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")

object_truth_df = object_truth_cat.get_quantities(
    quantities=[
        "cosmodc2_id_truth",
        "id_truth",
        "objectId",
        "match_objectId",
        "truth_type",
        "ra_truth",
        "dec_truth",
        "redshift_truth",
        "flux_u_truth",
        "flux_g_truth",
        "flux_r_truth",
        "flux_i_truth",
        "flux_z_truth",
        "flux_y_truth",
        "mag_u_truth",
        "mag_g_truth",
        "mag_r_truth",
        "mag_i_truth",
        "mag_z_truth",
        "mag_y_truth",
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
    ],
    native_filters=[tract_filter([3634, 3635, 3636, 3827, 3828, 3829, 3830, 4025, 4026, 4027])],
)
object_truth_df = pd.DataFrame(object_truth_df)

max_ra = np.nanmax(object_truth_df["ra_truth"])
min_ra = np.nanmin(object_truth_df["ra_truth"])
max_dec = np.nanmax(object_truth_df["dec_truth"])
min_dec = np.nanmin(object_truth_df["dec_truth"])
ra_dec_filters = [f"ra >= {min_ra}", f"ra <= {max_ra}", f"dec >= {min_dec}", f"dec <= {max_dec}"]

vertices = hp.ang2vec(
    np.array([min_ra, max_ra, max_ra, min_ra]),
    np.array([min_dec, min_dec, max_dec, max_dec]),
    lonlat=True,
)
ipix = hp.query_polygon(32, vertices, inclusive=True)
healpix_filter = GCRQuery((lambda h: np.isin(h, ipix, assume_unique=True), "healpix_pixel"))

object_truth_df = object_truth_df[object_truth_df["truth_type"] == 1]

object_truth_df.drop_duplicates(subset=["cosmodc2_id_truth"], inplace=True)


print("Loading CosmoDC2...\n")  # noqa: WPS421

config_overwrite = {"catalog_root_dir": "/data/scratch/dc2_nfs/cosmoDC2_v1.1.4"}
cosmo_cat = GCRCatalogs.load_catalog("desc_cosmodc2", config_overwrite)

cosmo_df = cosmo_cat.get_quantities(
    quantities=[
        "galaxy_id",
        "ra",
        "dec",
        "ellipticity_1_true",
        "ellipticity_2_true",
        "ellipticity_1_true_dc2",
        "ellipticity_2_true_dc2",
        "shear_1",
        "shear_2",
        "convergence",
    ],
    filters=ra_dec_filters,
    native_filters=healpix_filter,
)
cosmo_df = pd.DataFrame(cosmo_df)


print("Merging...\n")  # noqa: WPS421

merge_df = object_truth_df.merge(
    cosmo_df, left_on="cosmodc2_id_truth", right_on="galaxy_id", how="left"
)

merge_df = merge_df[~merge_df["galaxy_id"].isna()]

merge_df.drop(columns=["ra_truth", "dec_truth"], inplace=True)

merge_df.rename(
    columns={
        "redshift_truth": "redshift",
        "flux_u_truth": "flux_u",
        "flux_g_truth": "flux_g",
        "flux_r_truth": "flux_r",
        "flux_i_truth": "flux_i",
        "flux_z_truth": "flux_z",
        "flux_y_truth": "flux_y",
        "mag_u_truth": "mag_u",
        "mag_g_truth": "mag_g",
        "mag_r_truth": "mag_r",
        "mag_i_truth": "mag_i",
        "mag_z_truth": "mag_z",
        "mag_y_truth": "mag_y",
    },
    inplace=True,
)


print("Saving...\n")  # noqa: WPS421

with open(file_path, "wb") as f:
    pkl.dump(merge_df, f)

print(f"Catalog has been saved at {file_path}")  # noqa: WPS421
