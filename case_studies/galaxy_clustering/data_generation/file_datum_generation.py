import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from bliss.cached_dataset import FileDatum
from bliss.catalog import FullCatalog

min_flux_for_loss = 0
DATA_PATH = "/nfs/turbo/lsa-regier/scratch/kapnadak/new_data"
CATALOGS_PATH = DATA_PATH / Path("catalogs")
IMAGES_PATH = DATA_PATH / Path("images")
FILE_DATA_PATH = DATA_PATH / Path("file_data")
if not os.path.exists(FILE_DATA_PATH):
    os.makedirs(FILE_DATA_PATH)
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
)
BANDS = ("g", "r", "i", "z")
N_CATALOGS_PER_FILE = 50


def main(**kwargs):
    image_size = int(kwargs.get("image_size", 1280))
    tile_size = int(kwargs.get("tile_size", 128))
    n_tiles = int(image_size / tile_size)
    data: List[FileDatum] = []
    catalog_counter = 0
    file_counter = 0

    for catalog_path in CATALOGS_PATH.glob("*.dat"):
        catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=COL_NAMES)

        catalog_dict = {}
        catalog_dict["plocs"] = torch.tensor([catalog[["X", "Y"]].to_numpy()])
        n_sources = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, axis=1)
        catalog_dict["n_sources"] = n_sources
        catalog_dict["galaxy_fluxes"] = torch.tensor(
            [catalog[["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z"]].to_numpy()]
        )
        catalog_dict["star_fluxes"] = torch.zeros_like(catalog_dict["galaxy_fluxes"])
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

        filename = catalog_path.stem
        image_bands = []
        for band in BANDS:
            fits_filepath = IMAGES_PATH / Path(f"{filename}_{band}.fits")
            # Should the ordering in the bands matter? It does here.
            with fits.open(fits_filepath) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                image_bands.append(torch.from_numpy(image_data))
        stacked_image = torch.stack(image_bands, dim=0)

        data.append(
            FileDatum(
                {
                    "tile_catalog": tile_catalog_dict,
                    "images": stacked_image,
                    "background": stacked_image,
                }
            )
        )
        catalog_counter += 1
        if catalog_counter == N_CATALOGS_PER_FILE:
            stackname = f"{FILE_DATA_PATH}/file_data_{file_counter}_size_{N_CATALOGS_PER_FILE}.pt"
            torch.save(data, stackname)
            file_counter += 1
            catalog_counter = 0
            data = []


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
