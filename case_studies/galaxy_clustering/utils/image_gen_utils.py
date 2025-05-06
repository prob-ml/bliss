# flake8: noqa
# pylint: skip-file
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import ascii as astro_ascii
from astropy.io import fits
from astropy.table import Table

DES_SVA_TILES = pd.read_pickle("/data/scratch/des/sva_map.pickle")
DES_DIR = Path(
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/"
)
PSF_DIR = Path("/nfs/turbo/lsa-regier/scratch/gapatron/psf-models/dr2_tiles")
DES_PIXEL_SCALE = 0.263
BAND_TO_COL = {"g": 5, "r": 6, "i": 7, "z": 8}
IMAGE_SIZE = 10000
NFILES = 1
GALSIM_PATH = (
    "/home/kapnadak/bliss/case_studies/galaxy_clustering"
    "/data_generation/custom-single-image-galsim.yaml"
)


def create_des_catalog(des_subdir, data_path, file_suffix):
    """Create Catalog from DES Data for a particular subdirectory.

    Args:
        des_subdir: DES Tile to process
        data_path: save directory
        file_suffix: suffix to add to filename
    """
    main_path = DES_DIR / Path(des_subdir) / Path(f"{des_subdir}_dr2_main.fits")
    flux_path = DES_DIR / Path(des_subdir) / Path(f"{des_subdir}_dr2_flux.fits")
    main_df = pd.DataFrame(fits.getdata(main_path))
    flux_df = pd.DataFrame(fits.getdata(flux_path))
    full_df = pd.merge(
        main_df, flux_df, left_on="COADD_OBJECT_ID", right_on="COADD_OBJECT_ID", how="left"
    )
    fluxes = np.array(
        full_df[
            [
                "FLUX_AUTO_G_x",
                "FLUX_AUTO_R_x",
                "FLUX_AUTO_I_x",
                "FLUX_AUTO_Z_x",
            ]
        ]
    )
    fluxes *= fluxes > 0
    hlrs = DES_PIXEL_SCALE * np.array(full_df["FLUX_RADIUS_R"])
    hlrs = 1e-4 + hlrs * (hlrs > 0)
    a = np.array(full_df["A_IMAGE"])
    b = np.array(full_df["B_IMAGE"])
    g = (a - b) / (a + b)
    g1 = g * np.cos(np.arctan(b / a))
    g2 = g * np.sin(np.arctan(b / a))

    mock_catalog = pd.DataFrame()
    mock_catalog["RA"] = np.array(full_df["ALPHAWIN_J2000_x"])
    mock_catalog["DEC"] = np.array(full_df["DEC_x"])
    mock_catalog["X"] = np.array(full_df["XWIN_IMAGE_R"])
    mock_catalog["Y"] = np.array(full_df["YWIN_IMAGE_R"])
    mock_catalog["MEM"] = 0
    mock_catalog["FLUX_G"] = fluxes[:, 0]
    mock_catalog["FLUX_R"] = fluxes[:, 1]
    mock_catalog["FLUX_I"] = fluxes[:, 2]
    mock_catalog["FLUX_Z"] = fluxes[:, 3]
    mock_catalog["HLR"] = hlrs
    mock_catalog["FRACDEV"] = 0
    mock_catalog["G1"] = g1
    mock_catalog["G2"] = g2
    mock_catalog["Z"] = 0
    mock_catalog["SOURCE_TYPE"] = 0

    catalog_dir = f"{data_path}/catalogs"
    if not os.path.exists(catalog_dir):
        os.makedirs(catalog_dir)
    filename = f"{data_path}/catalogs/catalog_{file_suffix}.dat"
    catalog_table = Table.from_pandas(mock_catalog)
    astro_ascii.write(catalog_table, filename, format="no_header", overwrite=True)


def galsim_render(des_subdir, data_path, band="g"):
    """Render Galsim image for specific band.

    Args:
        des_subdir: DES Tile to process
        data_path: save directory
        band: band to process, one of [g,r,i,z]
    """
    input_dir = f"{data_path}/catalogs"
    output_dir = f"{data_path}/images"
    output_ext = f"{des_subdir}/{des_subdir}_{band}.fits"
    psf_dir_subdir = PSF_DIR / Path(des_subdir)
    psf_file = [f for f in os.listdir(psf_dir_subdir) if f.endswith(f"{band}_nobkg.psf")][0]
    flux_col = BAND_TO_COL[band]
    args = []
    args.append("galsim")
    args.append(f"{GALSIM_PATH}")
    args.append(f"variables.image_size={IMAGE_SIZE}")
    args.append(f"variables.nfiles={NFILES}")
    args.append(f"variables.input_dir={input_dir}")
    args.append(f"variables.input_file=catalog_{des_subdir}.dat")
    args.append(f"variables.output_dir={output_dir}")
    args.append(f"variables.output_file={output_ext}")
    args.append(f"variables.psf_dir={psf_dir_subdir}")
    args.append(f"variables.psf_file={psf_file}")
    args.append(f"variables.flux_col={flux_col}")
    subprocess.run(args, shell=False, check=False)
