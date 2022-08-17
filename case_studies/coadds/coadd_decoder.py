from pathlib import Path
from typing import Dict, Optional

import galsim
import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from bliss.catalog import FullCatalog
from bliss.datasets.background import ConstantBackground
from bliss.datasets.galsim_galaxies import GalsimBlends
from bliss.models.galsim_decoder import (
    SingleGalsimGalaxyDecoder,
    SingleGalsimGalaxyPrior,
    UniformGalsimGalaxiesPrior,
)
from case_studies.coadds.align_single_exposures import align_single_exposures


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _linear_coadd(aligned_images, weight):
    num = torch.sum(torch.mul(weight, aligned_images), dim=0)
    return num / torch.sum(weight, dim=0)


def _load_psf_from_file(psf_image_file: str, pixel_scale: float) -> galsim.GSObject:
    """Return normalized PSF galsim.GSObject from numpy psf_file."""
    assert Path(psf_image_file).suffix == ".npy"
    psf_image = np.load(psf_image_file)
    assert len(psf_image.shape) == 3 and psf_image.shape[0] == 1
    psf_image = galsim.Image(psf_image[0], scale=pixel_scale)
    return galsim.InterpolatedImage(psf_image).withFlux(1.0)


class CoaddUniformGalsimGalaxiesPrior(UniformGalsimGalaxiesPrior):
    def __init__(
        self,
        single_galaxy_prior: SingleGalsimGalaxyPrior,
        max_n_sources: int,
        max_shift: float,
        n_dithers: int,
    ):
        super().__init__(
            single_galaxy_prior,
            max_n_sources,
            max_shift,
        )
        self.n_dithers = n_dithers

    def sample(self) -> Dict[str, Tensor]:
        """Returns a single batch of source parameters."""
        d = super().sample()
        d["dithers"] = torch.distributions.uniform.Uniform(-0.5, 0.5).sample([self.n_dithers, 2])
        return d


class CoaddSingleGalaxyDecoder(SingleGalsimGalaxyDecoder):
    def __init__(
        self,
        slen: int,
        n_bands: int,
        pixel_scale: float,
        psf_image_file: str,
    ):
        super().__init__(
            slen,
            n_bands,
            pixel_scale,
            psf_image_file,
        )
        assert n_bands == 1, "Only 1 band is supported"
        self.slen = slen
        self.n_bands = 1
        self.pixel_scale = pixel_scale
        self.psf = _load_psf_from_file(psf_image_file, self.pixel_scale)

    def render_galaxy(
        self,
        galaxy_params: Tensor,
        psf: galsim.GSObject,
        slen: int,
        offset: Optional[Tensor] = None,
        dithers: Optional[Tensor] = None,
    ) -> Tensor:
        assert offset is None or offset.shape == (2,)
        if isinstance(galaxy_params, Tensor):
            galaxy_params = galaxy_params.cpu().detach()
        total_flux, disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b = galaxy_params
        bulge_frac = 1 - disk_frac

        disk_flux = total_flux * disk_frac
        bulge_flux = total_flux * bulge_frac

        components = []
        if disk_flux > 0:
            b_d = a_d * disk_q
            disk_hlr_arcsecs = np.sqrt(a_d * b_d)
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
                q=disk_q,
                beta=beta_radians * galsim.radians,
            )
            components.append(disk)
        if bulge_flux > 0:
            b_b = bulge_q * a_b
            bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs
            ).shear(q=bulge_q, beta=beta_radians * galsim.radians)
            components.append(bulge)
        galaxy = galsim.Add(components)
        gal_conv = galsim.Convolution(galaxy, psf)
        offset = (0, 0) if offset is None else offset
        dithers = (0, 0) if dithers is None else dithers
        shift = torch.add(torch.Tensor(dithers), torch.Tensor(offset))
        shift = shift.reshape(1, 2) if len(shift) == 2 else shift
        images = []
        for i in shift:
            image = gal_conv.drawImage(
                nx=slen, ny=slen, method="auto", scale=self.pixel_scale, offset=i
            )
            image = image.array
            images.append(image)
        return torch.tensor(images[:]).reshape(len(shift), 1, slen, slen)


class CoaddFullCatalogDecoder:
    def __init__(
        self,
        single_galaxy_decoder: CoaddSingleGalaxyDecoder,
        slen: int,
        bp: int,
        dithers: Optional[Tensor],
    ) -> None:
        self.single_decoder = single_galaxy_decoder
        self.slen = slen
        self.bp = bp
        self.dithers = dithers
        assert self.slen + 2 * self.bp >= self.single_decoder.slen

    def __call__(self, full_cat: FullCatalog):
        return self.render_catalog(full_cat, self.single_decoder.psf, self.dithers)

    def render_catalog(
        self, full_cat: FullCatalog, psf: galsim.GSObject, dithers: Optional[Tensor]
    ):
        size = self.slen + 2 * self.bp
        full_plocs = full_cat.plocs
        b, max_n_sources, _ = full_plocs.shape
        assert b == 1, "Only one batch supported for now."
        assert self.single_decoder.n_bands == 1, "Only 1 band supported for now"

        image = torch.zeros(len(dithers), 1, size, size)
        image0 = torch.zeros(1, 1, size, size)
        noiseless_centered = torch.zeros(len(dithers), max_n_sources, 1, size, size)
        noiseless_uncentered = torch.zeros(len(dithers), max_n_sources, 1, size, size)

        n_sources = int(full_cat.n_sources[0].item())
        galaxy_params = full_cat["galaxy_params"][0]
        plocs = full_plocs[0]

        for ii in range(n_sources):
            offset_x = plocs[ii][1] + self.bp - size / 2
            offset_y = plocs[ii][0] + self.bp - size / 2
            offset = torch.tensor([offset_x, offset_y])
            centered = self.single_decoder.render_galaxy(
                galaxy_params[ii], psf, size, dithers=dithers
            )
            uncentered = self.single_decoder.render_galaxy(galaxy_params[ii], psf, size, offset)
            uncentered_dithered = self.single_decoder.render_galaxy(
                galaxy_params[ii], psf, size, offset, dithers
            )
            noiseless_centered[:, ii] = centered.reshape(centered.shape[0], 1, size, size)
            noiseless_uncentered[:, ii] = uncentered.reshape(uncentered.shape[0], 1, size, size)
            image0 += uncentered
            image += uncentered_dithered
        return image, noiseless_centered, noiseless_uncentered, image0


class CoaddGalsimBlends(GalsimBlends):
    """Dataset of coadd galsim blends."""

    def __init__(
        self,
        prior: CoaddUniformGalsimGalaxiesPrior,
        decoder: CoaddFullCatalogDecoder,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
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
        noiseless, noiseless_centered, noiseless_uncentered, image0 = self.decoder.render_catalog(
            full_cat=full_cat, psf=self.decoder.single_decoder.psf, dithers=dithers
        )
        aligned_images = align_single_exposures(
            image0.reshape(size, size), noiseless, size, dithers
        )
        background = self.background.sample(rearrange(aligned_images, "d h w -> d 1 h w").shape)
        aligned_images = rearrange(aligned_images, "d h w -> d 1 h w")
        weight = 1 / (aligned_images + background.clone().detach())
        noisy_aligned_image = _add_noise_and_background(aligned_images, background)
        coadded_image = _linear_coadd(noisy_aligned_image, weight)
        return noiseless, noiseless_centered, noiseless_uncentered, background, coadded_image

    def _add_metrics(self, full_cat: FullCatalog, *args):
        return full_cat

    def __getitem__(self, idx):
        full_cat, dithers = self._sample_full_catalog()
        noiseless, _, _, background, coadded_image = self._get_images(full_cat, dithers)
        full_cat = self._add_metrics(full_cat, noiseless, background, coadded_image)
        tile_params = self._get_tile_params(full_cat)
        return {
            "noiseless": noiseless,
            "background": background,
            "images": coadded_image,
            **tile_params,
        }
