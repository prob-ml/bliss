from typing import Optional

import galsim
import torch
from torch import Tensor

from bliss.datasets.background import ConstantBackground
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.models.galsim_decoder import FullCatalogDecoder, UniformGalsimPrior
from case_studies.psf_homogenization.homogenization import psf_homo


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


class HomoGalsimBlends(GalsimBlends):
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
        self._size = self.slen + 2 * self.bp
        self.std_psf = self._get_gauss_psf(std_psf_fwhm)

    def _get_gauss_psf(self, fwhm):
        psf_obj = galsim.Gaussian(fwhm=fwhm)
        psf = psf_obj.drawImage(nx=self._size, ny=self._size, scale=self.pixel_scale)
        return torch.from_numpy(psf.array).reshape(1, 1, self._size, self._size)

    def _get_images(self, full_cat):
        noiseless, _, _ = self.decoder.render_catalog(full_cat)

        # get background and noisy image
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        # homogenization
        psf_obj = self.decoder.single_galaxy_decoder.psf
        psf = psf_obj.drawImage(nx=self._size, ny=self._size, scale=self.pixel_scale).array
        psf = torch.from_numpy(psf).reshape(1, 1, self._size, self._size)

        noisy_image = noisy_image.reshape(1, *noisy_image.shape)
        background = noisy_image.reshape(1, *noisy_image.shape)
        homo_image, _ = psf_homo(noisy_image, psf, self.std_psf, background)
        homo_image = homo_image.reshape(*noisy_image.shape)

        return noisy_image, background, homo_image

    def __getitem__(self, idx):
        full_cat = self._sample_full_catalog()
        images, background, homo_image = self._get_images(full_cat)
        tile_params = self._get_tile_params(full_cat)
        return {
            "images": homo_image,
            "background": background,
            "noisy_image": images,
            **tile_params,
        }
