# flake8: noqa
# pylint: skip-file
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from astropy.io import ascii as astro_ascii
from astropy.io import fits
from astropy.table import Table

from bliss.catalog import FullCatalog
from case_studies.galaxy_clustering.data_generation.prior import BackgroundPrior, ClusterPrior

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
    "GI_COLOR",
    "IZ_COLOR",
)

# ============================== Generate Catalogs ==============================


def catalog_gen(cfg):
    nfiles = int(cfg.nfiles)
    image_size = int(cfg.image_size)
    data_dir = cfg.data_dir
    catalogs_path = f"{data_dir}/catalogs/"
    file_prefix = "galsim_des"
    if not os.path.exists(catalogs_path):
        os.makedirs(catalogs_path)
    cluster_prior = ClusterPrior(image_size=image_size)
    background_prior = BackgroundPrior(image_size=image_size)

    for i in range(nfiles):
        file_name = f"{catalogs_path}/{file_prefix}_{i:03}.dat"
        if os.path.exists(file_name) and not cfg.overwrite:
            print(f"File {file_name} already exists, moving on ...")
            continue
        background_catalog = background_prior.sample_background()
        if np.random.uniform() < 0.5:
            cluster_catalog = cluster_prior.sample_cluster()
            catalog = pd.concat([cluster_catalog, background_catalog])
        else:
            catalog = background_catalog
        print(f"Writing catalog {i} ...")
        source_spread_x = (catalog["X"].max(), catalog["X"].min())
        source_spread_y = (catalog["Y"].max(), catalog["Y"].min())
        print(f"Cluster spread (X): {source_spread_x}")
        print(f"Cluster spread (Y): {source_spread_y}")
        print("\n")
        catalog_table = Table.from_pandas(catalog)
        astro_ascii.write(catalog_table, file_name, format="no_header", overwrite=True)


# ============================== Generate Images ==============================


def image_gen(cfg):
    image_size = cfg.image_size
    nfiles = cfg.nfiles
    data_dir = cfg.data_dir
    input_dir = f"{data_dir}/catalogs"
    output_dir = f"{data_dir}/images"
    args = []
    args.append("galsim")
    args.append("galsim-des.yaml")
    args.append(f"variables.image_size={image_size}")
    args.append(f"variables.nfiles={nfiles}")
    args.append(f"variables.input_dir={input_dir}")
    args.append(f"variables.output_dir={output_dir}")
    subprocess.run(args, shell=False, check=False)


# ============================== Generate File Datums ==============================


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
    n_tiles = int(image_size / tile_size)
    data: List[Dict] = []
    catalog_counter = 0
    file_counter = 0

    for catalog_path in catalogs_path.glob("*.dat"):
        catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=COL_NAMES)

        catalog_dict = {}
        catalog_dict["plocs"] = torch.tensor([catalog[["X", "Y"]].to_numpy()])
        catalog_dict["n_sources"] = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, axis=1)
        catalog_dict["fluxes"] = torch.tensor(
            [catalog[["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]].to_numpy()]
        )
        catalog_dict["membership"] = torch.tensor([catalog[["MEM"]].to_numpy()])
        catalog_dict["redshift"] = torch.tensor([catalog[["Z"]].to_numpy()])
        catalog_dict["galaxy_params"] = torch.tensor(
            [catalog[["HLR", "G1", "G2", "FRACDEV"]].to_numpy()]
        )
        catalog_dict["source_type"] = torch.ones_like(catalog_dict["membership"])
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
            fits_filepath = images_path / Path(f"{catalog_path.stem}_{band}.fits")
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


# ============================== CLI ==============================


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    catalog_gen(cfg.data_gen)
    # image_gen(cfg.data_gen)
    # file_data_gen(cfg.data_gen)


if __name__ == "__main__":
    main()
