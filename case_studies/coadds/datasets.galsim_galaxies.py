from typing import Dict, List, Optional

import galsim
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.background import ConstantBackground
from bliss.datasets.sdss import convert_flux_to_mag
from bliss.models.galsim_decoder import (
    FullCatalogDecoder,
    SingleGalsimGalaxyDecoder,
    SingleGalsimGalaxyPrior,
    UniformGalsimGalaxiesPrior,
)
from bliss.reporting import get_single_galaxy_ellipticities

def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise

class CoaddGalsimBlends(GalsimBlends):
    """Dataset of coadd galsim blends."""

    def __init__(self,
        prior: UniformGalsimGalaxiesPrior,
        decoder: FullCatalogDecoder,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.background = background
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = valid_n_batches

        # images
        self.max_n_sources = self.prior.max_n_sources
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.bp = self.decoder.bp
        self.slen = self.decoder.slen
        self.pixel_scale = self.decoder.single_decoder.pixel_scale

    def _get_images(self, full_cat):
        noiseless, noiseless_centered, noiseless_uncentered = self.decoder(full_cat)

        # get background and noisy image.
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        return noisy_image, noiseless, noiseless_centered, noiseless_uncentered, background



