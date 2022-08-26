from typing import Dict

import torch
from einops import rearrange
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.models.galsim_decoder import (
    FullCatalogDecoder,
    SingleGalsimGalaxyPrior,
    UniformGalsimPrior,
)
from case_studies.coadds.align_single_exposures import align_single_exposures


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _linear_coadd(aligned_images, weight):
    num = torch.sum(torch.mul(weight, aligned_images), dim=0)
    return num / torch.sum(weight, dim=0)


class CoaddUniformGalsimPrior(UniformGalsimPrior):
    def __init__(
        self,
        single_galaxy_prior: SingleGalsimGalaxyPrior,
        max_n_sources: int,
        max_shift: float,
        galaxy_prob: float,
        n_dithers: int,
    ):
        super().__init__(
            single_galaxy_prior,
            max_n_sources,
            max_shift,
            galaxy_prob,
        )
        self.n_dithers = n_dithers

    def sample(self) -> Dict[str, Tensor]:
        """Returns a single batch of source parameters."""
        d = super().sample()
        d["dithers"] = torch.distributions.uniform.Uniform(-0.5, 0.5).sample([self.n_dithers, 2])
        return d


class CoaddFullCatalogDecoder(FullCatalogDecoder):
    def render_catalog(self, full_cat: FullCatalog, dithers: Tensor):
        size = self.slen + 2 * self.bp
        images = torch.zeros(len(dithers), 1, size, size)
        plocs = full_cat.plocs.clone()
        image0, _, _ = super().render_catalog(full_cat)
        for ii, dth in enumerate(dithers):
            full_cat.plocs = plocs[0, :] + dth.reshape(1, 2)
            image, _, _ = super().render_catalog(full_cat)
            images[ii] = image
        return images, image0


class CoaddGalsimBlends(GalsimBlends):
    """Dataset of coadd galsim blends."""

    def _sample_full_catalog(self):
        params_dict = self.prior.sample()
        dithers = params_dict["dithers"]
        params_dict.pop("dithers")
        params_dict["plocs"] = params_dict["locs"] * self.slen
        params_dict.pop("locs")
        params_dict = {k: v.unsqueeze(0) for k, v in params_dict.items()}
        return FullCatalog(self.slen, self.slen, params_dict), dithers

    def _get_images(self, full_cat, dithers):
        size = self.slen + 2 * self.bp
        noiseless, image0 = self.decoder.render_catalog(full_cat, dithers)
        aligned_images = align_single_exposures(
            image0.reshape(size, size), noiseless, size, dithers
        )
        background = self.background.sample(rearrange(aligned_images, "d h w -> d 1 h w").shape)
        aligned_images = rearrange(aligned_images, "d h w -> d 1 h w")
        weight = 1 / (aligned_images + background.clone().detach())
        noisy_aligned_image = _add_noise_and_background(aligned_images, background)
        coadded_image = _linear_coadd(noisy_aligned_image, weight)
        return noiseless, coadded_image, background

    def __getitem__(self, idx):
        full_cat, dithers = self._sample_full_catalog()
        noiseless, coadded_image, background = self._get_images(full_cat, dithers)
        tile_params = self._get_tile_params(full_cat)
        return {
            "images": coadded_image,
            "noiseless": noiseless,
            "background": background,
            **tile_params,
        }
