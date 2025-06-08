import subprocess
import h5py
import pandas as pd
import os
from pathlib import Path
import numpy as np
import torch
from astropy.io import fits

BAND_TO_COL = {"g": 5, "r": 6, "i": 7, "z": 8}
GALSIM_PATH = "/data/scratch/gapatron/galaxy_clustering/desdr1_galsim/config/galsim_config.yaml"
PSF_DIR = Path("/data/scratch/gapatron/galaxy_clustering/desdr1_galsim/psf-models")

def load_data_native_endian(h5_group, include_keys_2d=None):
    data = {}
    include_keys_2d = include_keys_2d or []

    for k in h5_group.keys():
        arr = h5_group[k][:]
        if arr.ndim == 1 or k in include_keys_2d:
            if arr.dtype.byteorder == '>':
                arr = arr.byteswap().newbyteorder()
            data[k] = arr
    return data




def read_cluster_catalog(
            cl_catalog_path: str = 'y3_redmapper_v6.4.22+2_release.h5',
        ):
    """Read the cluster catalog."""
    with h5py.File(cl_catalog_path, 'r') as file:
          members_data = load_data_native_endian(
          file["catalog"]["cluster_members"],
                    include_keys_2d=["model_mag", "model_magerr"]
                    )
          cluster_data = load_data_native_endian(file["catalog"]["cluster"])
    members_data_1d = {k:v for k,v in members_data.items() if v.ndim == 1}
    members_data_2d = {k:v for k,v in members_data.items() if v.ndim == 2}
    members_df = pd.DataFrame(members_data_1d)
    for col in ["model_mag", "model_magerr"]:
            if col in members_data_2d:
                    bands = ['g', 'r', 'i', 'z']
                    expanded = pd.DataFrame(members_data_2d[col], columns=[f"{col}_{b}" for b in bands])
                    members_df = pd.concat([members_df, expanded], axis=1)
    cluster_indices = pd.unique(members_df["mem_match_id"])
    cluster_catalog = pd.DataFrame(cluster_data)
    return members_df, cluster_catalog, cluster_indices



def galsim_render_band(
        band,
        des_subdir,
        data_path,
        cat_path,
        nfiles=1,
        image_size=10000,
        psf_filepath=PSF_DIR,
        galsim_confpath=GALSIM_PATH,
    ):

    """Render Galsim image for DES tile.

    Args:
        des_subdir: DES Tile to process
        data_path: save directory. Must be at the root of catalogs, config and images.
    """
    output_filename = f"{des_subdir}_{band}.fits.fz"
    output_ext = f"{data_path}/images/{des_subdir}/{output_filename}"

    psf_subdir = f"{psf_filepath}/{des_subdir}"
    psf_file = [f for f in os.listdir(psf_subdir) if f.endswith(f"_{band}.psf")][0]

    os.makedirs(os.path.dirname(output_ext), exist_ok=True)
    print(f"[{des_subdir}-{band}] Using PSF file: {psf_file}")
    print(f"{output_ext} will be saved to {os.path.dirname(output_ext)}")
    print(os.listdir(os.path.dirname(output_ext)))
    if output_filename in os.listdir(os.path.dirname(output_ext)):
        print(f"[{des_subdir}-{band}] Image already exists, return path instead.")
        return output_ext  # Return the path to the rendered image

    args = [
        "galsim", galsim_confpath,
        f"eval_variables.image_size={image_size}",
        f"eval_variables.nfiles={nfiles}",
        f"eval_variables.input_file={cat_path}",
        f"eval_variables.output_file={output_ext}",
        f"eval_variables.psf_dir={psf_subdir}",
        f"eval_variables.psf_file={psf_file}",
        f"eval_variables.flux_col={BAND_TO_COL[band]}"
    ]

    subprocess.run(args, shell=False, check=True)

    return output_ext  # Return the path to the rendered image
