# flake8: noqa
# Utilities
import os
from pathlib import Path
import math
from typing import Dict, List
from tqdm import tqdm
from astropy.io import ascii as astro_ascii
from astropy.io import fits
from astropy.table import Table
from case_studies.galaxy_clustering.data_generation.prior import Prior
from bliss.catalog import FullCatalog
from case_studies.galaxy_clustering.data_generation.gen_utils import run_band, check_existing_and_load_meta, slice_and_save_subtiles
import numpy as np
import pandas as pd
import torch
import hydra
import multiprocessing
import csv
import time
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor


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

def process_single_tile_catalog(args):
    des_subdir, cfg = args
    data_path = cfg.data_dir
    catalog_dir = f"{data_path}/catalogs"
    os.makedirs(catalog_dir, exist_ok=True)

    filename = f"{catalog_dir}/{des_subdir}.cat"
    if os.path.exists(filename):
        print(f"[SKIP] Catalog exists for {des_subdir}", flush=True)
        return

    try:
        print(f"[START] Processing tile {des_subdir}", flush=True)
        prior = Prior(
            image_size=cfg.image_size,
            load_cluster_catalog=True
        )
        des_catalog = prior.make_des_catalog(
            des_subdir=des_subdir,
            class_star_thr=cfg.class_star_thr,
        )
        catalog_table = Table.from_pandas(des_catalog)
        astro_ascii.write(catalog_table, filename, format="no_header", overwrite=True)
        print(f"[DONE] {des_subdir}", flush=True)
    except Exception as e:
        print(f"[ERROR] {des_subdir}: {e}", flush=True)

def catalog_gen(cfg):
    data_path = cfg.data_dir
    os.makedirs(data_path, exist_ok=True)

    # Parallel over SLURM tasks
    rank = int(os.environ.get("SLURM_PROCID", 0))
    n_ranks = int(os.environ.get("SLURM_NTASKS", 1))

    all_subdirs = sorted(os.listdir(cfg.desdr_subdirs))
    des_subdirs = all_subdirs[rank::n_ranks]
    print(f"[INFO] Rank {rank}/{n_ranks} processing {len(des_subdirs)} tiles", flush=True)

    args = [(d, cfg) for d in des_subdirs]
    max_workers = min(len(des_subdirs), 4)  # Adjust this as needed

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_single_tile_catalog, args)

    print("Catalog generation completed.", flush=True)

def image_gen(cfg, des_subdir, catalog_path):
    timeout_seconds = 600
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

### subtile data generation

# def file_data_gen(cfg, des_subdir=None):
#     des_image_size = int(cfg.des_image_size)
#     image_size = int(cfg.image_size)
#     tile_size = int(cfg.tile_size)
#     data_dir = cfg.data_dir
#     min_flux_for_loss = float(cfg.min_flux_for_loss)
#     catalogs_path = Path(f"{data_dir}/catalogs/")
#     file_dir = Path(f"{data_dir}/file_data/")
#     file_dir.mkdir(parents=True, exist_ok=True)

#     n_tiles = math.ceil(image_size / tile_size)
#     subtile_size = 4
#     stride = 3
#     n_subtiles_per_tile = ((n_tiles - subtile_size) // stride + 1) ** 2

#     file_counter = 0

#     # 🔄 Subdir list handling
#     des_subdirs = [des_subdir] if des_subdir else sorted(os.listdir(cfg.desdr_subdirs))

#     for d in tqdm(des_subdirs, desc="Processing subdirs"):
#         result = check_existing_and_load_meta(
#             file_dir, catalogs_path, d,
#             n_subtiles_per_tile, des_image_size, image_size, tile_size
#         )

#         if result is None:
#             continue

#         if result["type"] == "full_file":
#             stacked_image, tile_catalog_dict = result["images"], result["tile_catalog"]

#         elif result["type"] == "catalog":
#             catalog, cropped_cat = result["catalog"], result["cropped_cat"]
#             stacked_image = image_gen(cfg, d, cropped_cat)
#             if stacked_image is None:
#                 print(f"Skipping {d} due to None image.")
#                 continue

#             catalog_dict = {
#                 "plocs": torch.from_numpy(catalog[["X", "Y"]].to_numpy()).unsqueeze(0),
#                 "fluxes": torch.from_numpy(catalog[["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]].to_numpy()).unsqueeze(0),
#                 "membership": torch.from_numpy(catalog[["MEM"]].to_numpy()).unsqueeze(0),
#                 "redshift": torch.from_numpy(catalog[["Z"]].to_numpy()).unsqueeze(0),
#                 "galaxy_params": torch.from_numpy(catalog[["HLR", "G1", "G2", "FRACDEV"]].to_numpy()).unsqueeze(0),
#                 "source_type": torch.from_numpy(catalog[["SOURCE_TYPE"]].to_numpy()).unsqueeze(0),
#             }
#             catalog_dict["n_sources"] = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, dim=1)

#             full_catalog = FullCatalog(height=image_size, width=image_size, d=catalog_dict)
#             tile_catalog = full_catalog.to_tile_catalog(tile_slen=tile_size, max_sources_per_tile=12 * tile_size)
#             tile_catalog = tile_catalog.filter_by_flux(min_flux=min_flux_for_loss)
#             tile_catalog = tile_catalog.get_brightest_sources_per_tile(band=2, exclude_num=0)

#             membership_array = np.zeros((n_tiles, n_tiles), dtype=bool)
#             for i, coords in enumerate(full_catalog["plocs"].squeeze()):
#                 if full_catalog["membership"][0, i, 0] > 0:
#                     tile_coord_y, tile_coord_x = (
#                         torch.div(coords, tile_size, rounding_mode="trunc").to(torch.int).tolist()
#                     )
#                     membership_array[tile_coord_x, tile_coord_y] = True

#             tile_catalog["membership"] = torch.tensor(membership_array).unsqueeze(0).unsqueeze(3).unsqueeze(4)
#             tile_catalog_dict = {key: torch.squeeze(val, 0) for key, val in tile_catalog.items()}

#         else:
#             continue

#         file_counter = slice_and_save_subtiles(
#             file_dir, stacked_image, tile_catalog_dict,
#             d, file_counter, n_tiles,
#             subtile_size, stride, tile_size, image_size
#         )


#### whole file data generation

# def file_data_gen(cfg, des_subdir=None):
#     des_image_size = int(cfg.des_image_size)
#     image_size = int(cfg.image_size)  # Expect 9728
#     tile_size = 512                   # <--- Changed from cfg.tile_size to 512
#     data_dir = cfg.data_dir
#     min_flux_for_loss = float(cfg.min_flux_for_loss)
#     catalogs_path = Path(f"{data_dir}/catalogs/")
#     file_dir = Path(f"{data_dir}/file_data/")  # <--- Output dir changed
#     file_dir.mkdir(parents=True, exist_ok=True)

#     n_tiles = image_size // tile_size  # <--- 9728 / 512 = 19
#     assert image_size % tile_size == 0

#     des_subdirs = [des_subdir] if des_subdir else sorted(os.listdir(cfg.desdr_subdirs))

#     for d in tqdm(des_subdirs, desc="Processing subdirs"):
#         result = check_existing_and_load_meta(
#             file_dir, catalogs_path, d,
#             None,  # <--- n_subtiles_per_tile not needed
#             des_image_size, image_size, tile_size
#         )
#         print(f"[START] {d}")

#         if result is None:
#             print(f"[SKIP] {d} - result is None")
#             continue

#         print(f"[TYPE] {d} - result type: {result['type']}")

#         if result["type"] == "catalog":
#             catalog, cropped_cat = result["catalog"], result["cropped_cat"]
#             stacked_image = image_gen(cfg, d, cropped_cat)
#             if stacked_image is None:
#                 print(f"[SKIP] {d} - image_gen returned None")
#                 continue


#         if result is None:
#             continue

#         if result["type"] == "full_file":
#             stacked_image, tile_catalog_dict = result["images"], result["tile_catalog"]

#         elif result["type"] == "catalog":
#             catalog, cropped_cat = result["catalog"], result["cropped_cat"]
#             stacked_image = image_gen(cfg, d, cropped_cat)
#             if stacked_image is None:
#                 print(f"Skipping {d} due to None image.")
#                 continue

#             catalog_dict = {
#                 "plocs": torch.from_numpy(catalog[["X", "Y"]].to_numpy()).unsqueeze(0),
#                 "fluxes": torch.from_numpy(catalog[["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]].to_numpy()).unsqueeze(0),
#                 "membership": torch.from_numpy(catalog[["MEM"]].to_numpy()).unsqueeze(0),
#                 "redshift": torch.from_numpy(catalog[["Z"]].to_numpy()).unsqueeze(0),
#                 "galaxy_params": torch.from_numpy(catalog[["HLR", "G1", "G2", "FRACDEV"]].to_numpy()).unsqueeze(0),
#                 "source_type": torch.from_numpy(catalog[["SOURCE_TYPE"]].to_numpy()).unsqueeze(0),
#             }
#             catalog_dict["n_sources"] = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, dim=1)

#             full_catalog = FullCatalog(height=image_size, width=image_size, d=catalog_dict)

#             # 💡 Compute 19x19 membership grid directly
#             membership_array = np.zeros((n_tiles, n_tiles), dtype=bool)
#             for i, coords in enumerate(full_catalog["plocs"].squeeze()):
#                 if full_catalog["membership"][0, i, 0] > 0:
#                     tile_coord_y, tile_coord_x = (
#                         torch.div(coords, tile_size, rounding_mode="trunc").to(torch.int).tolist()
#                     )
#                     if 0 <= tile_coord_x < n_tiles and 0 <= tile_coord_y < n_tiles:
#                         membership_array[tile_coord_x, tile_coord_y] = True

#             membership_tensor = torch.tensor(membership_array).unsqueeze(0).unsqueeze(3).unsqueeze(4)  # [1, 19, 19, 1, 1]
#             tile_catalog_dict = {"membership": membership_tensor}

#         else:
#             continue

#         # ✅ Save one file per full tile — no subtiles
#         save_path = file_dir / f"file_data_{d}_imagesize_{image_size}_tilesize_{tile_size}.pt"
#         torch.save({
#             "stacked_image": stacked_image,
#             "membership": tile_catalog_dict["membership"]
#         }, save_path)


### membership only file data generation

def file_data_gen(cfg, des_subdir=None):
    des_image_size = int(cfg.des_image_size)
    image_size = int(cfg.image_size)  # Expect 9728
    tile_size = 512
    data_dir = cfg.data_dir
    catalogs_path = Path(f"{data_dir}/catalogs/")
    file_dir = Path(f"{data_dir}/file_data/")
    file_dir.mkdir(parents=True, exist_ok=True)
    print("here")
    n_tiles = image_size // tile_size
    assert image_size % tile_size == 0

    des_subdirs = [des_subdir] if des_subdir else sorted(os.listdir(cfg.desdr_subdirs))

    for d in tqdm(des_subdirs, desc="Processing subdirs"):
        result = check_existing_and_load_meta(
            file_dir, catalogs_path, d,
            None,  # <--- n_subtiles_per_tile not needed
            des_image_size, image_size, tile_size
        )
        print(f"[START] {d}")

        if result is None:
            print(f"[SKIP] {d} - result is None")
            continue

        print(f"[TYPE] {d} - result type: {result['type']}")

        if result["type"] == "full_file":
            stacked_image, tile_catalog_dict = result["images"], result["tile_catalog"]

        elif result["type"] == "catalog":
            catalog, cropped_cat = result["catalog"], result["cropped_cat"]
            stacked_image = None  # <-- Skip image generation

            catalog_dict = {
                "plocs": torch.from_numpy(catalog[["X", "Y"]].to_numpy()).unsqueeze(0),
                "membership": torch.from_numpy(catalog[["MEM"]].to_numpy()).unsqueeze(0),
            }

            membership_array = np.zeros((n_tiles, n_tiles), dtype=bool)
            for i, coords in enumerate(catalog_dict["plocs"].squeeze()):
                if catalog_dict["membership"][0, i, 0] > 0:
                    tile_coord_y, tile_coord_x = (
                        torch.div(coords, tile_size, rounding_mode="trunc").to(torch.int).tolist()
                    )
                    if 0 <= tile_coord_x < n_tiles and 0 <= tile_coord_y < n_tiles:
                        membership_array[tile_coord_x, tile_coord_y] = True

            membership_tensor = torch.tensor(membership_array).unsqueeze(0).unsqueeze(3).unsqueeze(4)
            tile_catalog_dict = {"membership": membership_tensor}

        else:
            continue

        save_path = file_dir / f"file_data_{d}_imagesize_{image_size}_tilesize_{tile_size}.pt"
        torch.save({
            "stacked_image": stacked_image,  # Will be None for now
            "membership": tile_catalog_dict["membership"]
        }, save_path)


def _parallel_image_gen(args):
    cfg, des_subdir = args
    catalog_path = f"{cfg.data_dir}/catalogs/{des_subdir}.cat"
    if not os.path.exists(catalog_path):
        print(f"Catalog not found for {des_subdir}, skipping.")
        return
    image_gen(cfg, des_subdir, catalog_path)

def _parallel_file_data_gen(args):
    cfg_data_gen, des_subdir = args
    # Override to only run on the specified subdir
    print(f"Generating file data for {des_subdir} ...")
    file_data_gen(cfg_data_gen, des_subdir=des_subdir)


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
    from multiprocessing import Pool
    from concurrent.futures import ProcessPoolExecutor
    import os

    rank = int(os.environ.get("SLURM_PROCID", 0))
    n_ranks = int(os.environ.get("SLURM_NTASKS", 1))
    multiprocessing.set_start_method("spawn", force=True)
    print("Starting data generation process ...")

    if cfg.data_gen.catalog_gen:
        print("Starting catalog generation process ...")
        catalog_gen(cfg.data_gen)
        print(" ... Done!")
    if cfg.data_gen.image_gen:
        print("Starting image generation process ...")

        #### multiple nodes
        # des_subdirs = sorted(os.listdir(cfg.data_gen.desdr_subdirs))
        all_subdirs = sorted(os.listdir(cfg.data_gen.desdr_subdirs))
        des_subdirs = all_subdirs[rank::n_ranks]

        n_workers = min(len(des_subdirs), 4)  # adjust as needed
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            executor.map(_parallel_image_gen, [(cfg.data_gen, d) for d in des_subdirs])
    if cfg.data_gen.file_data_gen:
        print("Starting file data generation process ...")

        all_subdirs = sorted(os.listdir(cfg.data_gen.desdr_subdirs))
        des_subdirs = all_subdirs[rank::n_ranks]
        n_workers = min(len(des_subdirs), 4)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            executor.map(_parallel_file_data_gen, [(cfg.data_gen, d) for d in des_subdirs])

        print(" ... Done!")


if __name__ == "__main__":
    main()
