# flake8: noqa
# Utilities
import os
from pathlib import Path
import math
from typing import Dict, List
# Generate DES DR1 catalogs for galaxy clustering
from astropy.io import ascii as astro_ascii
from astropy.io import fits
from astropy.table import Table
from case_studies.galaxy_clustering.data_generation.prior import Prior
# Render Galsim images for galaxy clustering
from bliss.catalog import FullCatalog
from case_studies.galaxy_clustering.data_generation.gen_utils import galsim_render_band
# Generate file data (with tile catalogs and images) for galaxy clustering
from data_gen import file_data_gen
import numpy as np
import pandas as pd
import torch
# Configuration management
import hydra
import multiprocessing

# Constants
DES_DIR = Path(
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/"
)
DES_PIXEL_SCALE = 0.263
BANDS = ("g", "r", "i", "z")

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



def catalog_gen(cfg):
    """Generate DES DR1 catalogs for galaxy clustering.
    Amounts to mapping DES DR1 columns to a galsim catalog format.
    It creates a catalog for each DES tile and saves it in the specified output directory.

    Args:
        cfg: configuration object containing parameters for catalog generation


    """
    data_path = cfg.data_dir
    des_subdirs = cfg.desdr_subdirs
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    prior = Prior(
            image_size=cfg.image_size,
            load_cluster_catalog=True # Catalog is hardcoded in the Prior class for now.
        )
    print(f"Generating catalogs for {len(os.listdir(des_subdirs))} DES tiles...")
    # How to lexicographically sort the subdirs?
    for i, des_subdir in enumerate(sorted(os.listdir(des_subdirs))):
        print(f"Processing tile {des_subdir}...")
        des_catalog = prior.make_des_catalog(
            des_subdir=des_subdir,
            class_star_thr=cfg.class_star_thr,
        )
        catalog_dir = f"{data_path}/catalogs"
        if not os.path.exists(catalog_dir):
            os.makedirs(catalog_dir)

        filename = f"{data_path}/catalogs/{des_subdir}.cat"
        catalog_table = Table.from_pandas(des_catalog)
        astro_ascii.write(catalog_table, filename, format="no_header", overwrite=True)
        print(f"Finished processing tile {i + 1}/{len(os.listdir(des_subdirs))}: {des_subdir}")
    print("Catalog generation completed.")


def image_gen(cfg):
    """Generate Galsim images for galaxy clustering.

    Args:
        cfg: configuration object containing parameters for image generation
    """
    data_path = cfg.data_dir
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    des_subdirs = sorted(os.listdir(DES_DIR))

    args_list = []

    for des_subdir in des_subdirs:
        args_list = [(
                band,
                des_subdir,
                cfg.data_dir,
                1,  # nfiles
                10000,  # image_size
                cfg.psf_model_path,
                cfg.galsim_confpath
            ) for band in BANDS]
        
        with multiprocessing.Pool(processes=4) as pool:
            pool.starmap(galsim_render_band, args_list)
        
    print("Image generation completed.")



def file_data_gen(cfg):
    image_size = int(cfg.image_size)
    tile_size = int(cfg.tile_size)
    data_dir = cfg.data_dir
    n_catalogs_per_file = int(cfg.n_catalogs_per_file)
    bands = cfg.bands
    min_flux_for_loss = float(cfg.min_flux_for_loss)
    catalogs_path = Path(f"{data_dir}/catalogs/")
    images_path = f"{data_dir}/images/"
    file_path = f"{data_dir}/file_data/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    n_tiles = math.ceil(image_size / tile_size)
    data: List[Dict] = []
    catalog_counter = 0
    file_counter = 0

    for catalog_path in sorted(catalogs_path.glob("*.cat")):
        catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=COL_NAMES)

        catalog_dict = {}
        catalog_dict["plocs"] = torch.from_numpy(catalog[["X", "Y"]].to_numpy()).unsqueeze(0)
        catalog_dict["n_sources"] = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, axis=1)
        catalog_dict["fluxes"] = torch.from_numpy(
            catalog[["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]].to_numpy()
        ).unsqueeze(0)
        catalog_dict["membership"] = torch.from_numpy(catalog[["MEM"]].to_numpy()).unsqueeze(0)
        catalog_dict["redshift"] = torch.from_numpy(catalog[["Z"]].to_numpy()).unsqueeze(0)
        catalog_dict["galaxy_params"] = torch.from_numpy(
            catalog[["HLR", "G1", "G2", "FRACDEV"]].to_numpy()
        ).unsqueeze(0)
        catalog_dict["source_type"] = torch.from_numpy(catalog[["SOURCE_TYPE"]].to_numpy()).unsqueeze(0)
        full_catalog = FullCatalog(height=image_size, width=image_size, d=catalog_dict)
        tile_catalog = full_catalog.to_tile_catalog(
            tile_slen=tile_size,
            max_sources_per_tile=12 * tile_size,
        )
        tile_catalog = tile_catalog.filter_by_flux(min_flux=min_flux_for_loss)
        tile_catalog = tile_catalog.get_brightest_sources_per_tile(band=2, exclude_num=0)

        membership_array = np.zeros((n_tiles, n_tiles), dtype=bool)
        for i, coords in enumerate(full_catalog["plocs"].squeeze()):
            if full_catalog["membership"][0, i, 0] > 0:
                tile_coord_y, tile_coord_x = (
                    torch.div(coords, tile_size, rounding_mode="trunc").to(torch.int).tolist()
                )
                membership_array[tile_coord_x, tile_coord_y] = True

        tile_catalog["membership"] = (
            torch.tensor(membership_array).unsqueeze(0).unsqueeze(3).unsqueeze(4)
        )

        tile_catalog_dict = {}
        for key, value in tile_catalog.items():
            tile_catalog_dict[key] = torch.squeeze(value, 0)

        image_bands = []
        for band in bands:
            fits_filepath = images_path / Path(f"{catalog_path.stem}") / Path(f"{catalog_path.stem}_{band}.fits")
            with fits.open(fits_filepath) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                image_bands.append(torch.from_numpy(image_data))
        stacked_image = torch.stack(image_bands, dim=0)

        data.append(
            {
                "tile_catalog": tile_catalog_dict,
                "images": stacked_image,
            }
        )
        catalog_counter += 1
        if catalog_counter == n_catalogs_per_file:
            stackname = f"{file_path}/file_data_{file_counter}_size_{n_catalogs_per_file}.pt"
            torch.save(data, stackname)
            file_counter += 1
            data, catalog_counter = [], 0


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    """Main function to generate DES DR1 catalogs, Galsim images, and file data for galaxy clustering.
    Args:
        cfg: configuration object containing parameters for data generation
    """
    # There are three main steps in the data generation process:
    # 1. Generate DES DR1 catalogs for galaxy clustering.
    # 2. Render Galsim images for galaxy clustering.
    # 3. Generate file data (with tile catalogs and images) for galaxy clustering.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    print("Starting data generation process ...")
    if cfg.data_gen.catalog_gen:
        print("Starting catalog generation process ...")
        catalog_gen(cfg.data_gen)
        print(" ... Done!")
    if cfg.data_gen.image_gen:
        print("Starting image generation process ...")
        image_gen(cfg.data_gen)
    if cfg.data_gen.file_data_gen:
        print("Starting file data generation process ...")
        file_data_gen(cfg.data_gen)
        print(" ... Done!")


if __name__ == "__main__":
    main()
