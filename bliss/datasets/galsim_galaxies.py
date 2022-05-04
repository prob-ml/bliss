from pathlib import Path
from typing import Optional

import galsim
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bliss.datasets.background import ConstantBackground


def load_psf_from_file(psf_image_file: str, pixel_scale: float):
    """Return normalized PSF galsim.GSObject from numpy psf_file."""
    assert Path(psf_image_file).suffix == ".npy"
    psf_image = np.load(psf_image_file)
    assert len(psf_image.shape) == 3 and psf_image.shape[0] == 1
    psf_image = galsim.Image(psf_image[0], scale=pixel_scale)
    return galsim.InterpolatedImage(psf_image).withFlux(1.0)


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


class GalsimGalaxyPrior:
    def __init__(
        self,
        flux_sample: str,
        min_flux: float,
        max_flux: float,
        a_sample: str,
        alpha: Optional[float] = None,
        min_a_d: Optional[float] = None,
        max_a_d: Optional[float] = None,
        min_a_b: Optional[float] = None,
        max_a_b: Optional[float] = None,
        a_concentration: Optional[float] = None,
        a_loc: Optional[float] = None,
        a_scale: Optional[float] = None,
        a_bulge_disk_ratio: Optional[float] = None,
    ) -> None:
        self.flux_sample = flux_sample
        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = alpha
        if self.flux_sample == "pareto":
            assert self.alpha is not None

        self.a_sample = a_sample
        self.min_a_d = min_a_d
        self.max_a_d = max_a_d
        self.min_a_b = min_a_b
        self.max_a_b = max_a_b

        self.a_concentration = a_concentration
        self.a_loc = a_loc
        self.a_scale = a_scale
        self.a_bulge_disk_ratio = a_bulge_disk_ratio

        if self.a_sample == "uniform":
            assert self.min_a_d is not None
            assert self.max_a_d is not None
            assert self.min_a_b is not None
            assert self.max_a_b is not None
        elif self.a_sample == "gamma":
            assert self.a_concentration is not None
            assert self.a_loc is not None
            assert self.a_scale is not None
            assert self.a_bulge_disk_ratio is not None
        else:
            raise NotImplementedError()

    def sample(self, total_latent, device):
        # create galaxy as mixture of Exponential + DeVacauleurs
        if self.flux_sample == "uniform":
            total_flux = self._uniform(self.min_flux, self.max_flux, n_samples=total_latent)
        elif self.flux_sample == "pareto":
            total_flux = self._draw_pareto_flux(n_samples=total_latent)
        else:
            raise NotImplementedError()
        disk_frac = self._uniform(0, 1, n_samples=total_latent)
        beta_radians = self._uniform(0, 2 * np.pi, n_samples=total_latent)
        disk_q = self._uniform(0, 1, n_samples=total_latent)
        bulge_q = self._uniform(0, 1, n_samples=total_latent)
        if self.a_sample == "uniform":
            disk_a = self._uniform(self.min_a_d, self.max_a_d, n_samples=total_latent)
            bulge_a = self._uniform(self.min_a_b, self.max_a_b, n_samples=total_latent)
        elif self.a_sample == "gamma":
            disk_a = self._gamma(
                self.a_concentration, self.a_loc, self.a_scale, n_samples=total_latent
            )
            bulge_a = self._gamma(
                self.a_concentration,
                self.a_loc / self.a_bulge_disk_ratio,
                self.a_scale / self.a_bulge_disk_ratio,
                n_samples=total_latent,
            )
        else:
            raise NotImplementedError()
        return torch.stack(
            [total_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a], dim=1
        )

    @staticmethod
    def _uniform(a, b, n_samples=1) -> Tensor:
        # uses pytorch to return a single float ~ U(a, b)
        return (a - b) * torch.rand(n_samples) + b

    def _draw_pareto_flux(self, n_samples=1) -> Tensor:
        # draw pareto conditioned on being less than f_max
        assert self.alpha is not None
        u_max = 1 - (self.min_flux / self.max_flux) ** self.alpha
        uniform_samples = torch.rand(n_samples) * u_max
        return self.min_flux / (1.0 - uniform_samples) ** (1 / self.alpha)

    @staticmethod
    def _gamma(concentration, loc, scale, n_samples=1):
        x = torch.distributions.Gamma(concentration, rate=1.0).sample((n_samples,))
        return x * scale + loc


class GalsimGalaxyDecoder:
    def __init__(
        self,
        slen,
        n_bands,
        pixel_scale,
        psf_image_file: str,
    ) -> None:

        self.slen = slen
        assert n_bands == 1, "Only 1 band is supported"
        self.n_bands = 1
        self.pixel_scale = pixel_scale
        self.psf = load_psf_from_file(psf_image_file, self.pixel_scale)

    def __call__(self, z: Tensor) -> Tensor:
        if z.shape[0] == 0:
            return torch.zeros(0, 1, self.slen, self.slen, device=z.device)
        images = []
        for latent in z:
            image = self.render_galaxy(latent)
            images.append(image)
        return torch.stack(images, dim=0).to(z.device)

    def render_galaxy(self, galaxy_params) -> Tensor:
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
        # convolve with PSF
        gal_conv = galsim.Convolution(galaxy, self.psf)
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale
        )
        return torch.from_numpy(image.array).reshape(1, self.slen, self.slen)


class SDSSGalaxies(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        prior: GalsimGalaxyPrior,
        decoder: GalsimGalaxyDecoder,
        background: ConstantBackground,
        num_workers: int,
        batch_size: int,
        n_batches: int,
    ):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.background = background

    def __getitem__(self, idx):
        galaxy_params = self.prior.sample(1, "cpu")
        galaxy_image = self.decoder.render_galaxy(galaxy_params[0])
        background = self.background.sample((1, *galaxy_image.shape)).squeeze(1)
        galaxy_with_background = galaxy_image + background
        noise = galaxy_with_background.sqrt() * torch.randn_like(galaxy_with_background)
        galaxy_with_noise = galaxy_with_background + noise
        snr = torch.sqrt(torch.sum(galaxy_image**2 / galaxy_with_background)).reshape(1)
        return {
            "images": galaxy_with_noise,
            "background": background,
            "noiseless": galaxy_image,
            "params": galaxy_params[0],
            "snr": snr,
        }

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)
