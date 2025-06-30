import subprocess
import h5py
import pandas as pd
import os
from pathlib import Path
import numpy as np
import torch
from astropy.io import fits
import math

BAND_TO_COL = {"g": 5, "r": 6, "i": 7, "z": 8}
GALSIM_PATH = "/data/scratch/gapatron/galaxy_clustering/desdr1_galsim/config/galsim_config.yaml"
PSF_DIR = Path("/data/scratch/gapatron/galaxy_clustering/desdr1_galsim/psf-models")
COL_NAMES = (
    "RA",
    "DEC",
    "X",
    "Y",
    "MEM",
    "FLUX_G",
    "FLUX_R",
    "FLUX_I",
    "FLUX_Z",
    "HLR",
    "FRACDEV",
    "G1",
    "G2",
    "Z",
    "SOURCE_TYPE",
    "PMEM",
)


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

def run_band(args, q):

    try:
        q.put(galsim_render_band(*args))
    except Exception as e:
        q.put(e)


def check_existing_and_load_meta(
        file_dir,
        catalogs_path,
        des_subdir,
        n_subtiles_per_tile,
        des_image_size,
        image_size,
        tile_size
):
    """Check if subtiles already exist, and load metadata if not.
    Args:
        file_dir: Directory where file data is stored.
        catalogs_path: Directory where catalogs are stored.
        des_subdir: DES tile subdirectory to process.
        n_subtiles_per_tile: Number of subtiles per tile.
        des_image_size: Desired image size for the tile.
        image_size: Image size to check against.
        tile_size: Size of the tile.
    """
    # Check if subtiles already exist
    for idx in range(n_subtiles_per_tile):
        path = file_dir / f"file_data_*_destile_{des_subdir}_imagesize_{image_size}_tilesize_{tile_size}_subtile_{idx}.pt"
        if not any(path.parent.glob(path.name)):
            break
    else:
        print(f"All subtiles for {des_subdir} already exist. Skipping.")
        return None

    # Try to load full file
    full_tile_path = next(file_dir.glob(f"file_data_*_destile_{des_subdir}_imagesize_{image_size}_size_1.pt"), None)
    if full_tile_path is not None:
        print(f"Found full file for {des_subdir}, slicing...")
        data = torch.load(full_tile_path)
        if isinstance(data, list): data = data[0]
        if data["images"] is None:
            print(f"Skipping {des_subdir} due to None image.")
            return None
        return {"type": "full_file", "images": data["images"], "tile_catalog": data["tile_catalog"]}

    # Load catalog for later image generation
    catalog_path = catalogs_path / f"{des_subdir}.cat"
    if not catalog_path.exists():
        print(f"Catalog for {des_subdir} does not exist, skipping.")
        return None

    catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=COL_NAMES)
    if des_image_size != image_size:
        offset = (des_image_size - image_size) // 2
        catalog["X"] -= offset
        catalog["Y"] -= offset
        catalog = catalog[
            (catalog["X"] >= 0) & (catalog["Y"] >= 0) &
            (catalog["X"] < image_size) & (catalog["Y"] < image_size)
        ]
        cropped_cat = catalogs_path / f"{des_subdir}_cropped_{image_size}.cat"
        catalog.to_csv(cropped_cat, sep=" ", index=False, header=False)
    else:
        cropped_cat = catalog_path

    return {"type": "catalog", "catalog": catalog, "cropped_cat": cropped_cat}


def slice_and_save_subtiles(
        file_dir,
        stacked_image,
        tile_catalog_dict,
        des_subdir,
        file_counter,
        n_tiles,
        subtile_size,
        stride,
        tile_size,
        image_size
):
    """
    Slice the stacked image into subtiles and save them with their metadata.

    Args:
        file_dir: Directory where file data is stored.
        stacked_image: Stacked image tensor to slice.
        tile_catalog_dict: Dictionary containing tile catalog metadata.
        des_subdir: DES tile subdirectory to process.
        file_counter: Counter for naming output files.
        n_tiles: Number of tiles in the stacked image.
        subtile_size: Size of the subtile.
        stride: Stride for sliding window.
        tile_size: Size of the tile.
        image_size: Size of the image.
    """
    subtile_idx = 0
    for i in range(0, n_tiles - subtile_size + 1, stride):
        for j in range(0, n_tiles - subtile_size + 1, stride):
            subtile_catalog = {
                key: val[i:i + subtile_size, j:j + subtile_size]
                for key, val in tile_catalog_dict.items()
            }
            y_start, x_start = i * tile_size, j * tile_size
            y_end, x_end = y_start + subtile_size * tile_size, x_start + subtile_size * tile_size
            subtile_image = stacked_image[:, y_start:y_end, x_start:x_end].clone()

            subtile_data = {
                "des_subdir": des_subdir,
                "tile_catalog": subtile_catalog,
                "images": subtile_image,
            }

            subtile_path = file_dir / f"file_data_{file_counter}_destile_{des_subdir}_imagesize_{image_size}_tilesize_{tile_size}_subtile_{subtile_idx}.pt"
            torch.save(subtile_data, subtile_path)

            subtile_idx += 1
            file_counter += 1

    return file_counter