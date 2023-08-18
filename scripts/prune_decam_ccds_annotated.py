import numpy as np
from astropy.io import fits
from astropy.table import Table

# Script to reduce the size of the ccds-annotated-decam file by only keeping fixed CCDs used for
# DECaLS PSF params. Use to regenerate data/tests/decals/ccds-annotated-decam-dr9-small.fits.

with open("/home/zhteoh/bliss/data/decals/ccds-annotated-decam-dr9.fits", "rb") as f:
    ccds_annotated_fits = fits.open(f)

ccds_annotated_table = Table.read("/home/zhteoh/bliss/data/decals/ccds-annotated-decam-dr9.fits")

BRICKNAME = "3366m010"
brick_ccds = Table.read(
    f"/home/zhteoh/bliss/data/decals/{BRICKNAME[:3]}/{BRICKNAME}/legacysurvey-{BRICKNAME}-ccds.fits"
)
fixed_ccds = brick_ccds[
    (
        brick_ccds["image_filename"]
        == "decam/CP/V4.8.2a/CP20141020/c4d_141021_015854_ooi_g_ls9.fits.fz"
    )
    | (
        brick_ccds["image_filename"]
        == "decam/CP/V4.8.2a/CP20151107/c4d_151108_003333_ooi_r_ls9.fits.fz"
    )
    | (
        brick_ccds["image_filename"]
        == "decam/CP/V4.8.2a/CP20130912/c4d_130913_040652_ooi_z_ls9.fits.fz"
    )
]
keep_ccds = fixed_ccds["ccdname"]

psf_cols = [
    col
    for col in ccds_annotated_table.colnames
    if col.startswith("psf") or col.startswith("gal") or col.startswith("gauss")
]
brick_ccds_mask = np.isin(ccds_annotated_table["ccdname"], keep_ccds)
keep_cols = psf_cols + ["ccdname", "filter"]
ccds_annotated_table_small = ccds_annotated_table[brick_ccds_mask][keep_cols]

# write ccds_annotated_table_small as astropy table to file
save_location = "/home/zhteoh/bliss/data/tests/decals/ccds-annotated-decam-dr9-small.fits"
with open(save_location, "wb") as f:
    ccds_annotated_table_small.write(f, format="fits")

# du -sh /home/zhteoh/bliss/data/decals/ccds-annotated-decam-dr9.fits => 3.8G
# du -sh /home/zhteoh/bliss/data/tests/decals/ccds-annotated-decam-dr9-small.fits => 59M
