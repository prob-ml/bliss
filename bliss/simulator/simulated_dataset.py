import warnings
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from bliss.simulator.decoder import ImageDecoder
from bliss.simulator.prior import CatalogPrior
from bliss.surveys.survey import Survey

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        survey: Survey,
        prior: CatalogPrior,
        tile_slen: int,
        n_batches: int,
        coadd_depth: int = 1,
        num_workers: int = 0,
        valid_n_batches: Optional[int] = None,
        fix_validation_set: bool = False,
    ):
        super().__init__()

        self.survey = survey
        survey.prepare_data()

        self.catalog_prior = prior
        self.catalog_prior.requires_grad_(False)

        self.image_decoder = ImageDecoder(
            tile_slen=tile_slen,
            psf=survey.psf,
            bands=survey.BANDS,
            background=self.survey.background,
            flux_calibration_dict=survey.flux_calibration_dict,
            ref_band=prior.reference_band,
        )

        self.n_batches = n_batches
        self.coadd_depth = coadd_depth
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

        self.image_ids = self.survey.image_ids()

    def randomized_image_ids(self, num_samples=1):
        """Get random image_id from loaded params."""
        n = np.random.randint(len(self.image_ids), size=(num_samples,), dtype=int)
        # reorder self.image_ids to match the order of the sampled indices
        return [self.image_ids[i] for i in n], n

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
        image_ids, image_id_indices = self.randomized_image_ids(self.catalog_prior.batch_size)
        images, psf_params = self.image_decoder.render_images(
            tile_catalog,
            image_ids,
            image_id_indices,
            self.coadd_depth,
        )
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
