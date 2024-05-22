import os
from pathlib import Path
from typing import List, TypedDict

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from bliss.catalog import FullCatalog, TileCatalog

TileCatalog.allowed_params.update(["membership", "fracdev", "g1g2"])

DATA_PATH = Path(os.getcwd()) / Path("data")
CATALOGS_PATH = DATA_PATH / Path("padded_catalogs")
IMAGES_PATH = DATA_PATH / Path("images")
FILE_DATA_PATH = DATA_PATH / Path("file_data")
if not os.path.exists(FILE_DATA_PATH):
    os.makedirs(FILE_DATA_PATH)
COL_NAMES = [
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
]
FILE_PREFIX = "galsim_des"
BANDS = ["g", "r", "i", "z"]
N_CATALOGS_PER_FILE = 2

FileDatum = TypedDict(
    "FileDatum",
    {
        "tile_catalog": TileCatalog,
        "images": torch.Tensor,
        "background": torch.Tensor,
        "psf_params": torch.Tensor,
    },
)

data: List[FileDatum] = []

for CATALOG_PATH in CATALOGS_PATH.glob("*.dat"):
    catalog = pd.read_csv(CATALOG_PATH, sep=" ", header=None, names=COL_NAMES)

    catalog_dict = dict()
    catalog_dict["plocs"] = torch.tensor([catalog[["X", "Y"]].to_numpy()])
    n_sources = torch.sum(catalog_dict["plocs"][:, :, 0] != 0, axis=1)
    catalog_dict["n_sources"] = n_sources
    catalog_dict["fluxes"] = torch.tensor(
        [catalog[["FLUX_R", "FLUX_G", "FLUX_I", "FLUX_Z"]].to_numpy()]
    )
    catalog_dict["membership"] = torch.tensor([catalog[["MEM"]].to_numpy()])
    catalog_dict["hlr"] = torch.tensor([catalog[["TSIZE"]].to_numpy()])
    catalog_dict["fracdev"] = torch.tensor([catalog[["FRACDEV"]].to_numpy()])
    catalog_dict["g1g2"] = torch.tensor([catalog[["G1", "G2"]].to_numpy()])

    full_catalog = FullCatalog(height=5000, width=5000, d=catalog_dict)
    tile_catalog = full_catalog.to_tile_catalog(tile_slen=4, max_sources_per_tile=10)

    filename = CATALOG_PATH.stem
    pad_file_prefix = f"{FILE_PREFIX}_padded_"
    index = filename[len(pad_file_prefix) :]
    image_bands = list()
    for band in BANDS:
        fits_filepath = IMAGES_PATH / Path(f"{FILE_PREFIX}_{index}_{band}.fits")
        # Should the ordering in the bands matter? It does here.
        with fits.open(fits_filepath) as hdul:
            image_data = hdul[0].data.astype(np.float32)
            image_bands.append(torch.from_numpy(image_data))
    stacked_image = torch.stack(image_bands, dim=0)

    data.append(FileDatum({"tile_catalog": tile_catalog, "images": stacked_image}))

chunks = [data[i : i + N_CATALOGS_PER_FILE] for i in range(0, len(data), N_CATALOGS_PER_FILE)]
for i, chunk in enumerate(chunks):
    torch.save(chunk, f"{DATA_PATH}/file_data/file_data_{i}.pt")
