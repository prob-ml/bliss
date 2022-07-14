from typing import Optional, Dict, List

import galsim
import torch
from torch import Tensor
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.sdss import convert_flux_to_mag
from bliss.reporting import get_single_galaxy_ellipticities
from bliss.datasets.background import ConstantBackground
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.models.galsim_decoder import UniformGalsimGalaxiesPrior, FullCatalogDecoder
from case_studies.psf_homogenization.homogenization import psf_homo
from case_studies.psf_homogenization.galsim_star import UniformGalsimPrior, FullCatelogDecoderSG


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise




class PsfSampler:
    def __init__(
        self,
        psf_rmin: float = 0.7,
        psf_rmax: float = 0.9,
    ) -> None:
        self.rmin = psf_rmin
        self.rmax = psf_rmax

    def sample(self) -> galsim.GSObject:
        # sample psf from galsim Gaussian distribution
        if self.rmin == self.rmax:
            fwhm = self.rmin
        elif self.rmin > self.rmax:
            raise ValueError("invalid argument!!!")
        else:
            fwhm = torch.distributions.uniform.Uniform(self.rmin, self.rmax).sample([1]).item()
        
        return galsim.Gaussian(fwhm=fwhm)


class GalsimBlendswithPSF(GalsimBlends):
    def __init__(
        self,
        prior: UniformGalsimGalaxiesPrior,
        decoder: FullCatalogDecoder,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        psf_sampler: PsfSampler,
        std_psf_fwhm: float = 1.0,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__(
            prior,
            decoder,
            background,
            tile_slen,
            max_sources_per_tile,
            num_workers,
            batch_size,
            n_batches,
            fix_validation_set,
            valid_n_batches,
        )
        self.slen = self.decoder.slen
        self.pixel_scale = self.decoder.single_decoder.pixel_scale

        self.std_psf = torch.from_numpy(
            galsim.Gaussian(fwhm=std_psf_fwhm)
            .drawImage(nx=self.slen, ny=self.slen, scale=self.pixel_scale)  # noqa: WPS348
            .array  # noqa: WPS348
        )
        self.psf = psf_sampler

    def _get_images(self, full_cat):
        psf_obj = self.psf.sample()
        noiseless, noiseless_centered, noiseless_uncentered = self.decoder.render_catalog(
            full_cat, psf_obj
        )

        # get background and noisy image
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        # homogenization
        std_psf = self.std_psf.reshape(1, 1, self.slen, self.slen)
        psf = torch.from_numpy(
            psf_obj.drawImage(nx=self.slen, ny=self.slen, scale=self.pixel_scale).array
        ).reshape(1, 1, self.slen, self.slen)
        homo_image, _ = psf_homo(
            noisy_image.reshape(1, *noisy_image.shape),
            psf,
            std_psf,
            background.reshape(1, *noisy_image.shape),
        )

        return (  # noqa: WPS227
            noisy_image,
            noiseless,
            noiseless_centered,
            noiseless_uncentered,
            background,
            homo_image.reshape(*noisy_image.shape),
            psf,
        )

    def __getitem__(self, idx):
        full_cat = self._sample_full_catalog()
        images, *metric_images, background, homo_image, psf = self._get_images(full_cat)
        full_cat = self._add_metrics(full_cat, *metric_images, background)
        tile_params = self._get_tile_params(full_cat)
        return {
            "images": homo_image,
            "background": background,
            "noisy_image": images,
            "psf": psf,
            "std_psf": self.std_psf,
            **tile_params,
        }

class GalsimBlendsRand(GalsimBlends):
    def __init__(
        self,
        prior: UniformGalsimPrior,
        decoder: FullCatalogDecoder,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        psf_sampler: PsfSampler,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__(
            prior,
            decoder,
            background,
            tile_slen,
            max_sources_per_tile,
            num_workers,
            batch_size,
            n_batches,
            fix_validation_set,
            valid_n_batches,
        )
        self.slen = self.decoder.slen
        self.pixel_scale = self.decoder.single_decoder.pixel_scale
        self.psf = psf_sampler
    
    def _get_images(self, full_cat):
        psf_obj = self.psf.sample()
        noiseless, noiseless_centered, noiseless_uncentered = self.decoder.render_catalog(
            full_cat, psf_obj
            )

        # get background and noisy image.
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        return noisy_image, noiseless, noiseless_centered, noiseless_uncentered, background








