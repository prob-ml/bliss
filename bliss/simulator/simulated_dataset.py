import warnings
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from bliss.simulator.decoder import Decoder
from bliss.simulator.prior import CatalogPrior

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        prior: CatalogPrior,
        decoder: Decoder,
        n_batches: int,
        num_workers: int = 0,
        valid_n_batches: Optional[int] = None,
        fix_validation_set: bool = False,
    ):
        super().__init__()

        self.catalog_prior = prior
        self.catalog_prior.requires_grad_(False)

        self.decoder = decoder

        self.n_batches = n_batches
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

    def get_batch(self):
        """Get a batch of simulated images.

        The images are simulated by first generating a tile catalog from the prior, followed
        by choosing a random background and PSF and using those to generate the image. By default,
        the same row, camcol, and field combination is used for the background, PSF, and flux ratios
        of a single simulated image.

        Returns:
            Dict: A dictionary of the simulated TileCatalog, (batch_size, bands, height, width)
            tensors for images and background, and a (batch_size, 1, 6) tensor for the psf params.
        """
        tile_catalog = self.catalog_prior.sample()
        images, psf_params = self.decoder.render_images(tile_catalog)
        return {
            "tile_catalog": tile_catalog,
            "images": images,
            "psf_params": psf_params,
        }

    def __iter__(self):
        with torch.no_grad():
            for _ in range(self.n_batches):
                yield self.get_batch()

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.fix_validation_set:
            valid: List[Dict[str, Tensor]] = []
            for _ in tqdm(range(self.valid_n_batches), desc="Generating fixed validation set"):
                valid.append(self.get_batch())
            num_workers = 0
        else:
            valid = self
            num_workers = self.num_workers
        return DataLoader(valid, batch_size=None, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)
