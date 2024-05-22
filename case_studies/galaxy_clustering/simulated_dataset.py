import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset, IterableDataset

from bliss.catalog import FullCatalog
from bliss.simulator.simulated_dataset import FileDatum

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class GalaxyClusterCachedSimulatedDataset(pl.LightningDataModule):
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
        survey_bands: List[str],
        padded_catalogs: bool = True,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cached_data_path = cached_data_path
        self.file_prefix = file_prefix
        self.survey_bands = survey_bands
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
            tile_catalog = full_catalog.to_tile_catalog(tile_slen=5, max_sources_per_tile=4)

            # Extract index from catalog filepath in preparation to read images
            filename = catalog_path.stem
            pad_file_prefix = (
                self.file_prefix + "_padded_" if self.padded_catalogs else catalog_path.stem
            )
            index = filename[len(pad_file_prefix) :]

            # Read and stack image bands
            image_bands = list()
            for band in self.survey_bands:
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

            self.data.append({"catalog": tile_catalog, "images": stacked_image})
        self.slices = self.parse_slices(splits, len(self.data))
    
    def _percent_to_idx(self, x, length):
        """Converts string in percent to an integer index."""
        return int(float(x.strip()) / 100 * length) if x.strip() else None

    def parse_slices(self, splits: str, length: int):
        slices = [slice(0, 0) for _ in range(3)]  # default to empty slice for each split
        for i, data_split in enumerate(splits.split("/")):
            # map "start_percent:stop_percent" to slice(start_idx, stop_idx)
            slices[i] = slice(*(self._percent_to_idx(val, length) for val in data_split.split(":")))
        return slices

    def read_file(self, filename: str) -> List[FileDatum]:
        with open(filename, "rb") as f:
            return torch.load(f)

    def __len__(self):
        return len(self.data[self.slices[0]])

    def __getitem__(self, idx):
        return self.data[idx]

    def train_dataloader(self):
        assert self.data, "No cached train data loaded; run `generate.py` first"
        assert len(self.data[self.slices[0]]) >= self.batch_size, (
            f"Insufficient cached data loaded; "
            f"need at least {self.batch_size} "
            f"but only have {len(self.data[self.slices[0]])}. Re-run `generate.py` with "
            f"different generation `train_n_batches` and/or `batch_size`."
        )
        # TODO: consider adding pixelwise Poisson noise to the *training* images on the fly
        # to reduce overfitting rather than adding noise while simulating training images.
        # (validation and test images should still be simulated with noise, though)
        return DataLoader(
            self.data[self.slices[0]],
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        assert self.data[self.slices[1]], "No cached validation data found; run `generate.py` first"
        return DataLoader(
            self.data[self.slices[1]], batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        assert self.data[self.slices[2]], "No cached test data found; run `generate.py` first"
        return DataLoader(
            self.data[self.slices[2]], batch_size=self.batch_size, num_workers=self.num_workers
        )
