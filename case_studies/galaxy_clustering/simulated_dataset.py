import warnings
from typing import Tuple, Dict, List
from pathlib import Path

from astropy.io import fits
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset

import pytorch_lightning as pl
from bliss.simulator.simulated_dataset import CachedSimulatedDataset


# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class GalaxyClusterCachedSimulatedDataset(CachedSimulatedDataset):
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

    def __init__(
        self,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        file_prefix: str,
        bands: List[str],
        padded_catalogs: bool = True,
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cached_data_path = cached_data_path
        self.file_prefix = file_prefix
        self.bands = bands
        self.padded_catalogs = padded_catalogs

        self.data: List[Dict] = list()

        # Assume cached image and padded catalogs exist,
        # and each has their own subdirectory under
        # cached_data_path named images and padded_catalogs
        # catalogs are stored as *.dat and image bands as *.fits.
        catalog_paths = Path(self.cached_data_path) / Path("padded_catalogs")
        for catalog_path in catalog_paths.glob("*.dat"):
            # Read catalog entries
            catalog = pd.read_csv(catalog_path, sep=" ", header=None, names=self.COL_NAMES)

            # Extract index from catalog filepath in preparation to read images
            filename = catalog_path.stem
            pad_file_prefix = (
                self.file_prefix + "_padded_" if self.padded_catalogs else catalog_path.stem
            )
            index = filename[len(pad_file_prefix) :]

            # Read and stack image bands
            image_bands = list()
            for band in self.bands:
                fits_filepath = (
                    Path(self.cached_data_path)
                    / Path("images")
                    / Path(self.file_prefix + "_" + index + "_" + band + ".fits")
                )
                # Should the ordering in the bands matter? It does here.
                with fits.open(fits_filepath) as hdul:
                    image_data = hdul[0].data.astype(np.float32)
                    image_bands.append(torch.from_numpy(image_data))
            stacked_image = torch.stack(image_bands, dim=0)

            self.data.append({"catalog": catalog, "images": stacked_image})
            self.slices = self.parse_slices(splits, len(self.data))
