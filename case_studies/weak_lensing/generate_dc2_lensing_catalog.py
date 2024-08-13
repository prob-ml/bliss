import os
import pickle as pkl

import GCRCatalogs
import healpy as hp
import numpy as np
import pandas as pd
from GCRCatalogs import GCRQuery

file_name = "dc2_lensing_catalog.pkl"
file_path = os.path.join("/data", "scratch", "dc2local", file_name)
file_already_populated = os.path.isfile(file_path)

if file_already_populated:
    raise FileExistsError(f"{file_path} already exists.")

print("Loading truth table...\n")  # noqa: WPS421
# truth table
GCRCatalogs.set_root_dir("/data/scratch/dc2_nfs/")
truth_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_truth")
truth_data = truth_cat.get_quantities(
    [
        "id",
        "cosmodc2_id",
        "ra",
        "dec",
        "match_objectId",
        "flux_u",
        "flux_g",
        "flux_r",
        "flux_i",
        "flux_z",
        "flux_y",
        "truth_type",
    ]
)
max_ra = np.nanmax(truth_data["ra"])
min_ra = np.nanmin(truth_data["ra"])
max_dec = np.nanmax(truth_data["dec"])
min_dec = np.nanmin(truth_data["dec"])
pos_filters = [f"ra >= {min_ra}", f"ra <= {max_ra}", f"dec >= {min_dec}", f"dec <= {max_dec}"]

vertices = hp.ang2vec(
    np.array([min_ra, max_ra, max_ra, min_ra]),
    np.array([min_dec, min_dec, max_dec, max_dec]),
    lonlat=True,
)
ipix = hp.query_polygon(32, vertices, inclusive=True)
healpix_filter = GCRQuery((lambda h: np.isin(h, ipix, assume_unique=True), "healpix_pixel"))
truth_data = pd.DataFrame(truth_data)
truth_data.drop(["ra", "dec"], axis=1, inplace=True)

# object table
print("Loading object table...\n")  # noqa: WPS421
config_overwrite = {"catalog_root_dir": "/data/scratch/dc2_nfs/cosmoDC2"}
cosmo_cat = GCRCatalogs.load_catalog("desc_cosmodc2", config_overwrite)
cosmo_data = cosmo_cat.get_quantities(
    quantities=[
        "galaxy_id",
        "shear_1",
        "shear_2",
        "convergence",
        "ra",
        "dec",
        "mag_true_r",
        "galaxy_id",
        "position_angle_true",
        "size_minor_disk_true",
        "size_disk_true",
        "size_minor_bulge_true",
        "size_bulge_true",
        "bulge_to_total_ratio_i",
        "redshift",
        "ellipticity_1_true",
        "ellipticity_2_true",
    ],
    filters=pos_filters,
    native_filters=healpix_filter,
)
cosm_dat = pd.DataFrame(cosmo_data)

# psf
print("Loading PSF parameters...\n")  # noqa: WPS421
match_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")
psf_params = match_cat.get_quantities(
    [
        "IxxPSF_pixel_g",
        "IxxPSF_pixel_z",
        "IxxPSF_pixel_r",
        "IxxPSF_pixel_i",
        "IxxPSF_pixel_u",
        "IxxPSF_pixel_y",
        "IyyPSF_pixel_g",
        "IyyPSF_pixel_z",
        "IyyPSF_pixel_r",
        "IyyPSF_pixel_i",
        "IyyPSF_pixel_u",
        "IyyPSF_pixel_y",
        "IxyPSF_pixel_g",
        "IxyPSF_pixel_z",
        "IxyPSF_pixel_r",
        "IxyPSF_pixel_i",
        "IxyPSF_pixel_u",
        "IxyPSF_pixel_y",
        "psf_fwhm_g",
        "psf_fwhm_z",
        "psf_fwhm_r",
        "psf_fwhm_i",
        "psf_fwhm_u",
        "psf_fwhm_y",
        "cosmodc2_id_truth",
    ]
)
psf = pd.DataFrame(psf_params)

# merge
print("Merging...\n")  # noqa: WPS421
cosmo_truth = cosm_dat.merge(truth_data, left_on="galaxy_id", right_on="cosmodc2_id", how="left")

merge_with_object = cosmo_truth.merge(
    psf, left_on="galaxy_id", right_on="cosmodc2_id_truth", how="left"
)

# save
with open(file_path, "wb") as f:
    pkl.dump(merge_with_object, f)
print(f"Catalog has been saved at {file_path}")  # noqa: WPS421
