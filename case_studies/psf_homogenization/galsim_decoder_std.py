from typing import Optional

import galsim
import torch
from torch import Tensor

from bliss.datasets.background import ConstantBackground
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.models.galsim_decoder import FullCatalogDecoder, UniformGalsimPrior
from case_studies.psf_homogenization.homogenization import psf_homo
from case_studies.psf_homogenization.psf_sampler import PsfSampler


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


class GalsimBlendswithPSF(GalsimBlends):
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
        std_psf_fwhm: float = 1.0,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
        psf_sampler: Optional[PsfSampler] = None,
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
            psf_sampler,
            fix_validation_set,
            valid_n_batches,
        )
        self.std_psf = torch.from_numpy(
            galsim.Gaussian(fwhm=std_psf_fwhm)
            .drawImage(nx=self.slen, ny=self.slen, scale=self.pixel_scale)  # noqa: WPS348
            .array  # noqa: WPS348
        )

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

        noiseless_images = noiseless, noiseless_centered, noiseless_uncentered
        homo_image = homo_image.reshape(*noisy_image.shape)

        return noisy_image, *noiseless_images, background, homo_image, psf

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
