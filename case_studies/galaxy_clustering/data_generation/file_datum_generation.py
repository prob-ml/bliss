import os
import sys
from pathlib import Path
from typing import List, TypedDict

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from bliss.catalog import FullCatalog, TileCatalog

# pylint: disable=duplicate-code

MIN_FLUX_THRESHOLD = 0
DATA_PATH = Path(os.getcwd()) / Path("data")
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
    "FLUX_R",
    "FLUX_G",
    "FLUX_I",
    "FLUX_Z",
    "TSIZE",
    "FRACDEV",
    "G1",
    "G2",
)
BANDS = ("g", "r", "i", "z")
N_CATALOGS_PER_FILE = 10

FileDatum = TypedDict(
    "FileDatum",
    {
        "tile_catalog": TileCatalog,
        "images": torch.Tensor,
        "background": torch.Tensor,
        "psf_params": torch.Tensor,
    },
)


def main(**kwargs):
    image_size = int(kwargs.get("image_size", 4800))
    tile_size = int(kwargs.get("tile_size", 4))
    data: List[FileDatum] = []

    for catalog_path in CATALOGS_PATH.glob("*.dat"):
        catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=COL_NAMES)

        catalog_dict = {}
        catalog_dict["plocs"] = torch.tensor([catalog[["X", "Y"]].to_numpy()])
        catalog_dict["plocs"][:, :, 1] = image_size - catalog_dict["plocs"][:, :, 1]
        n_sources = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, axis=1)
        catalog_dict["n_sources"] = n_sources
        catalog_dict["galaxy_fluxes"] = torch.tensor(
            [catalog[["FLUX_R", "FLUX_G", "FLUX_I", "FLUX_Z"]].to_numpy()]
        )
        catalog_dict["star_fluxes"] = torch.zeros_like(catalog_dict["galaxy_fluxes"])
        catalog_dict["membership"] = torch.tensor([catalog[["MEM"]].to_numpy()])
        catalog_dict["galaxy_params"] = torch.tensor(
            [catalog[["TSIZE", "G1", "G2", "FRACDEV"]].to_numpy()]
        )
        catalog_dict["source_type"] = torch.ones_like(catalog_dict["membership"])
        full_catalog = FullCatalog(height=image_size, width=image_size, d=catalog_dict)
        tile_catalog = full_catalog.to_tile_catalog(
            tile_slen=tile_size,
            max_sources_per_tile=12 * tile_size,
        )
        tile_catalog = tile_catalog.filter_tile_catalog_by_flux(min_flux=MIN_FLUX_THRESHOLD)
        tile_catalog = tile_catalog.get_brightest_sources_per_tile(band=2, exclude_num=0)

        tile_catalog_dict = tile_catalog.to_dict()
        for key, value in tile_catalog_dict.items():
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

    chunks = [data[i : i + N_CATALOGS_PER_FILE] for i in range(0, len(data), N_CATALOGS_PER_FILE)]
    for i, chunk in enumerate(chunks):
        torch.save(chunk, f"{DATA_PATH}/file_data/file_data_{i}.pt")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
