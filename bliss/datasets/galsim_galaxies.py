from pathlib import Path
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
    GalsimGalaxiesDecoder,
    SingleGalsimGalaxyDecoder,
    SingleGalsimGalaxyPrior,
    UniformGalsimGalaxiesPrior,
)
from bliss.reporting import get_single_galaxy_ellipticities


class SingleGalsimGalaxies(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        prior: SingleGalsimGalaxyPrior,
        decoder: SingleGalsimGalaxyDecoder,
        background: ConstantBackground,
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

    def __getitem__(self, idx):
        galaxy_params = self.prior.sample(1, "cpu")
        galaxy_image = self.decoder.render_galaxy(galaxy_params[0])
        background = self.background.sample((1, *galaxy_image.shape)).squeeze(1)
        return {
            "images": _add_noise_and_background(galaxy_image, background),
            "background": background,
            "noiseless": galaxy_image,
            "params": galaxy_params[0],
            "snr": _get_snr(galaxy_image, background),
        }

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        dl = DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)
        if not self.fix_validation_set:
            return dl
        valid: List[Dict[str, Tensor]] = []
        for _ in tqdm(range(self.valid_n_batches), desc="Generating fixed validation set"):
            valid.append(next(iter(dl)))
        return DataLoader(valid, batch_size=None, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)


class GalsimBlends(SingleGalsimGalaxies):
    """Dataset of galsim blends."""

    def __init__(
        self,
        prior: UniformGalsimGalaxiesPrior,
        decoder: GalsimGalaxiesDecoder,
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
            num_workers,
            batch_size,
            n_batches,
            fix_validation_set,
            valid_n_batches,
        )

        # images
        self.max_n_sources = self.prior.max_n_sources
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.bp = self.decoder.bp
        self.slen = self.decoder.slen
        self.pixel_scale = self.decoder.pixel_scale

    def _compute_metrics(
        self,
        n_sources: int,
        galaxy_params: Tensor,
        noiseless: Tensor,
        noiseless_centered: Tensor,
        noiseless_uncentered: Tensor,
        background: Tensor,
    ):
        scale = self.pixel_scale
        size = self.slen + 2 * self.bp
        psf = self.decoder.psf
        psf_tensor = torch.from_numpy(psf.drawImage(nx=size, ny=size, scale=scale).array)

        single_galaxy_tensor = noiseless_centered[:n_sources]
        single_galaxy_tensor = rearrange(single_galaxy_tensor, "n 1 h w -> n h w", n=n_sources)
        mags = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            mags[ii] = convert_flux_to_mag(galaxy_params[ii, 0])
        ellips = torch.zeros(self.max_n_sources, 2)
        meas = get_single_galaxy_ellipticities(single_galaxy_tensor, psf_tensor, scale)
        ellips[:n_sources, :] = meas["ellips"]

        # get snr and blendedness
        snr = torch.zeros(self.max_n_sources)
        blendedness = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            snr[ii] = _get_snr(noiseless_centered[ii], background)
            blendedness[ii] = _get_blendedness(noiseless_uncentered[ii], noiseless)

        return mags, galaxy_params[:, 0], ellips, snr, blendedness

    def __getitem__(self, idx):
        # prepare parameters
        params_dict = self.prior.sample()
        params_dict["plocs"] = params_dict["locs"] * self.slen
        full_cat = FullCatalog(self.slen, self.slen, params_dict)
        n_sources = full_cat.n_sources.item()
        galaxy_params = full_cat["galaxy_params"]

        # forward images
        noiseless, noiseless_centered, noiseless_uncentered = self.decoder.forward(full_cat)
        noiseless = rearrange(noiseless, "b c h w -> c h w", b=1, c=1, h=self.slen + 2 * self.bp)
        noiseless_centered = rearrange(noiseless_centered, "b n c h w -> n c h w", b=1, c=1)
        noiseless_uncentered = rearrange(noiseless_uncentered, "b n c h w -> n c h w", b=1, c=1)

        # get background and noisy image.
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        mags, gal_fluxes, ellips, snr, blendedness = self._compute_metrics(
            n_sources,
            galaxy_params,
            noiseless,
            noiseless_centered,
            noiseless_uncentered,
            background,
        )

        # add to full catalog (needs to be in batches)
        full_cat["mags"] = rearrange(mags, "n 1 -> 1 n 1", n=self.max_n_sources)
        full_cat["fluxes"] = torch.zeros(1, self.max_n_sources, 1)  # stars
        full_cat["log_fluxes"] = torch.zeros(1, self.max_n_sources, 1)  # stars
        full_cat["galaxy_fluxes"] = rearrange(gal_fluxes, "n 1 -> 1 n 1", n=self.max_n_sources)

        full_cat["ellips"] = rearrange(ellips, "n g -> 1 n g", n=self.max_n_sources, g=2)
        full_cat["snr"] = snr.reshape(1, self.max_n_sources, 1)
        full_cat["blendedness"] = blendedness.reshape(1, self.max_n_sources, 1)

        # finally get corresponding tile catalog.
        tile_cat = full_cat.to_tile_params(self.tile_slen, self.max_sources_per_tile)

        return {
            "images": noisy_image,
            "background": background,
            **{k: v.reshape(self.max_n_sources, -1) for k, v in tile_cat.to_dict()},
        }


class ToyGaussian(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        num_workers,
        batch_size,
        n_batches,
        slen,
        n_bands,
        pixel_scale,
        background,
        psf_fwhm,
        min_flux,
        max_flux,
        min_hlr,
        max_hlr,
        max_e,
    ):
        super().__init__()
        assert n_bands == 1, "Only 1 band is supported"
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_batches = n_batches

        self.slen = slen
        self.n_bands = n_bands
        self.pixel_scale = pixel_scale

        # create background
        self.background = torch.zeros((self.n_bands, self.slen, self.slen), dtype=torch.float32)
        self.background[...] = background

        # small dummy psf
        self.psf = galsim.Gaussian(fwhm=psf_fwhm).withFlux(1.0)
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.min_hlr = min_hlr
        self.max_hlr = max_hlr
        self.max_e = max_e

    def _uniform(self, a, b):
        # uses pytorch to return a single float ~ U(a, b)
        unif = (a - b) * torch.rand(1) + b
        return unif.item()

    def __getitem__(self, idx):
        flux_avg = self._uniform(self.min_flux, self.max_flux)
        hlr = self._uniform(self.min_hlr, self.max_hlr)  # arcseconds
        flux = (hlr / self.pixel_scale) ** 2 * np.pi * flux_avg  # approx

        # sample ellipticity
        ell = self._uniform(0, self.max_e)
        theta = self._uniform(0, 2 * np.pi)
        g1 = ell * np.cos(theta)
        g2 = ell * np.sin(theta)

        # pylint: disable=no-value-for-parameter
        galaxy = galsim.Gaussian(half_light_radius=hlr).shear(g1=g1, g2=g2).withFlux(flux)
        gal_conv = galsim.Convolution(galaxy, self.psf)
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale
        )

        # convert image to pytorch and reshape
        image = torch.from_numpy(image.array).reshape(1, self.slen, self.slen)

        # add noise and background.
        image += self.background
        noise = torch.sqrt(image) * torch.randn(*image.shape)
        image += noise

        return {"images": image, "background": self.background}

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)


def load_psf_from_file(psf_image_file: str, pixel_scale: float):
    """Return normalized PSF galsim.GSObject from numpy psf_file."""
    assert Path(psf_image_file).suffix == ".npy"
    psf_image = np.load(psf_image_file)
    assert len(psf_image.shape) == 3 and psf_image.shape[0] == 1
    psf_image = galsim.Image(psf_image[0], scale=pixel_scale)
    return galsim.InterpolatedImage(psf_image).withFlux(1.0)


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _get_snr(image: Tensor, background: Tensor) -> float:
    image_with_background = image + background
    return torch.sqrt(torch.sum(image**2 / image_with_background)).item()


def _get_blendedness(single_galaxy: Tensor, all_galaxies: Tensor) -> float:
    num = torch.sum(single_galaxy * single_galaxy).item()
    denom = torch.sum(single_galaxy * all_galaxies).item()
    return 1 - num / denom
