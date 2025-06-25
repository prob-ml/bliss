# flake8: noqa
# Utilities
import os
from pathlib import Path
import math
from typing import Dict, List
from tqdm import tqdm
# Generate DES DR1 catalogs for galaxy clustering
from astropy.io import ascii as astro_ascii
from astropy.io import fits
from astropy.table import Table
from case_studies.galaxy_clustering.data_generation.prior import Prior
# Render Galsim images for galaxy clustering
from bliss.catalog import FullCatalog
from case_studies.galaxy_clustering.data_generation.gen_utils import galsim_render_band
# Generate file data (with tile catalogs and images) for galaxy clustering
import numpy as np
import pandas as pd
import torch
# Configuration management
import hydra
import multiprocessing
import csv
import time
from multiprocessing import Process, Queue


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
    for i, des_subdir in tqdm(enumerate(sorted(os.listdir(des_subdirs)))):
        print(f"Processing tile {des_subdir}...")
        des_catalog = prior.make_des_catalog(
            des_subdir=des_subdir,
            class_star_thr=cfg.class_star_thr,
        )
        catalog_dir = f"{data_path}/catalogs"
        if not os.path.exists(catalog_dir):
            os.makedirs(catalog_dir)

        filename = f"{data_path}/catalogs/{des_subdir}.cat"
        if os.path.exists(filename):
            print(f"Catalog for {des_subdir} already exists, skipping.")
            continue
        catalog_table = Table.from_pandas(des_catalog)
        astro_ascii.write(catalog_table, filename, format="no_header", overwrite=True)
        print(f"Finished processing tile {i + 1}/{len(os.listdir(des_subdirs))}: {des_subdir}")
    print("Catalog generation completed.")

def run_band(args, q):
            try:
                q.put(galsim_render_band(*args))
            except Exception as e:
                q.put(e)

def image_gen(cfg, des_subdir, catalog_path):
    timeout_seconds = 180
    band_paths = []
    processes = []
    queues = []

    for band in BANDS:
        args = (
            band, des_subdir, cfg.data_dir, catalog_path,
            1, cfg.image_size, cfg.psf_model_path, cfg.galsim_confpath
        )
        q = Queue()

        p = Process(target=run_band, args=(args, q))
        p.start()
        processes.append((p, q))
        queues.append(q)

    start = time.time()
    while time.time() - start < timeout_seconds:
        if all(not p.is_alive() for p, _ in processes):
            break
        time.sleep(1)

    if any(p.is_alive() for p, _ in processes):
        print(f"Tile {des_subdir} exceeded {timeout_seconds} seconds. Timing out and skipping.")
        for p, _ in processes:
            p.terminate()
            p.join()
        with open("timeout_tiles.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if not os.path.exists("timeout_tiles.csv"):
                writer.writerow(["tile"])
            writer.writerow([des_subdir])
        return None

    for q in queues:
        result = q.get()
        if isinstance(result, Exception):
            with open("timeout_tiles.csv", "a", newline='') as f:
                writer = csv.writer(f)
                if not os.path.exists("timeout_tiles.csv"):
                    writer.writerow(["tile"])
                writer.writerow([des_subdir])
            return None
        band_paths.append(result)

    print(f"Finished processing tile: {des_subdir}")
    images = [fits.getdata(p).astype(np.float32) for p in band_paths]
    return torch.stack([torch.from_numpy(im) for im in images])



def file_data_gen(cfg):
    des_image_size = int(cfg.des_image_size)
    image_size = int(cfg.image_size)
    tile_size = int(cfg.tile_size)
    data_dir = cfg.data_dir
    min_flux_for_loss = float(cfg.min_flux_for_loss)
    catalogs_path = Path(f"{data_dir}/catalogs/")
    file_dir = Path(f"{data_dir}/file_data/")
    file_dir.mkdir(parents=True, exist_ok=True)
    n_tiles = math.ceil(image_size / tile_size)
    subtile_size = 4
    stride = 3
    n_subtiles_per_tile = ((n_tiles - subtile_size) // stride + 1) ** 2

    # Helper function to check if subtile files already exist
    def subtile_file_exists(des_subdir):
        for idx in range(n_subtiles_per_tile):
            path = file_dir / f"file_data_*_destile_{des_subdir}_imagesize_{image_size}_tilesize_{tile_size}_subtile_{idx}.pt"
            if not any(path.parent.glob(path.name)):
                return False
        return True

    for des_subdir in tqdm(sorted(os.listdir(cfg.desdr_subdirs))):
        # Check if the subtiles already exist
        if subtile_file_exists(des_subdir):
            print(f"All subtiles for {des_subdir} already exist. Skipping.")
            continue

        # Look for full file instead
        full_tile_path = next(file_dir.glob(f"file_data_*_destile_{des_subdir}_imagesize_{image_size}_size_1.pt"), None)
        if full_tile_path is not None:
            print(f"Found full file for {des_subdir}, slicing into subtiles...")
            data = torch.load(full_tile_path)
            if isinstance(data, list):
                data = data[0]
            stacked_image = data["images"]
            if stacked_image is None:
                print(f"Skipping {des_subdir} due to None image.")
                continue
            tile_catalog_dict = data["tile_catalog"]
        else:
            catalog_path = catalogs_path / f"{des_subdir}.cat"
            if not catalog_path.exists():
                print(f"Catalog for {des_subdir} does not exist, skipping.")
                continue

            catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=COL_NAMES)
            if des_image_size != image_size:
                offset = (des_image_size - image_size) // 2
                catalog["X"] -= offset
                catalog["Y"] -= offset
                catalog = catalog[(catalog["X"] >= 0) & (catalog["Y"] >= 0) &
                                  (catalog["X"] < image_size) & (catalog["Y"] < image_size)]
                cropped_cat = catalogs_path / f"{des_subdir}_cropped_{image_size}.cat"
                catalog.to_csv(cropped_cat, sep=" ", index=False, header=False)
            else:
                cropped_cat = catalog_path

            stacked_image = image_gen(cfg, des_subdir, cropped_cat)
            if stacked_image is None:
                print(f"Skipping {des_subdir} due to None image.")
                continue

            catalog_dict = {
                "plocs": torch.from_numpy(catalog[["X", "Y"]].to_numpy()).unsqueeze(0),
                "fluxes": torch.from_numpy(catalog[["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]].to_numpy()).unsqueeze(0),
                "membership": torch.from_numpy(catalog[["MEM"]].to_numpy()).unsqueeze(0),
                "redshift": torch.from_numpy(catalog[["Z"]].to_numpy()).unsqueeze(0),
                "galaxy_params": torch.from_numpy(catalog[["HLR", "G1", "G2", "FRACDEV"]].to_numpy()).unsqueeze(0),
                "source_type": torch.from_numpy(catalog[["SOURCE_TYPE"]].to_numpy()).unsqueeze(0),
            }
            catalog_dict["n_sources"] = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, dim=1)

            full_catalog = FullCatalog(height=image_size, width=image_size, d=catalog_dict)
            tile_catalog = full_catalog.to_tile_catalog(tile_slen=tile_size, max_sources_per_tile=12 * tile_size)
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
            tile_catalog_dict = {key: torch.squeeze(val, 0) for key, val in tile_catalog.items()}

        # Slice into subtiles
        subtile_idx = 0
        for i in range(0, n_tiles - subtile_size + 1, stride):
            for j in range(0, n_tiles - subtile_size + 1, stride):
                subtile_catalog = {
                    key: val[i:i + subtile_size, j:j + subtile_size]
                    for key, val in tile_catalog_dict.items()
                }
                y_start = i * tile_size
                y_end = y_start + subtile_size * tile_size
                x_start = j * tile_size
                x_end = x_start + subtile_size * tile_size
                subtile_image = stacked_image[:, y_start:y_end, x_start:x_end]

                subtile_data = {
                    "des_subdir": des_subdir,
                    "tile_catalog": subtile_catalog,
                    "images": subtile_image,
                }

                subtile_path = file_dir / f"file_data_{file_counter}_destile_{des_subdir}_imagesize_{image_size}_tilesize_{tile_size}_subtile_{subtile_idx}.pt"
                torch.save(subtile_data, subtile_path)
                subtile_idx += 1

        file_counter += 1


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
