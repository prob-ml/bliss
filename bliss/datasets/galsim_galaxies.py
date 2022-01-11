from pathlib import Path

import galsim
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bliss.models.galaxy_net import OneCenteredGalaxyAE


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
        num_workers=0,
        batch_size=64,
        n_batches=10,
        slen=53,
        n_bands=1,
        pixel_scale=0.396,
        noise_factor=0.05,
        background=845,
        psf_fwhm=1.4,
        min_flux=300,
        max_flux=10000,
        min_hlr=0.8,
        max_hlr=4.0,
        max_e=0.6,
    ):
        super().__init__()
        assert n_bands == 1, "Only 1 band is supported"
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_batches = n_batches

        self.slen = slen
        self.n_bands = n_bands
        self.pixel_scale = pixel_scale
        self.noise_factor = noise_factor

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
        l = self._uniform(0, self.max_e)
        theta = self._uniform(0, 2 * np.pi)
        g1 = l * np.cos(theta)
        g2 = l * np.sin(theta)

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
        noise = torch.sqrt(image) * torch.randn(*image.shape) * self.noise_factor
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


class SDSSGalaxies(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        num_workers=0,
        batch_size=64,
        n_batches=10,
        n_bands=1,
        slen=53,
        noise_factor=1.0,
        min_flux=1e3,
        max_flux=3.5e5,
        min_a_d=0.8,
        max_a_d=6.5,
        min_a_b=0.8,
        max_a_b=3.6,
        background=865.0,
        pixel_scale=0.396,  # SDSS
        psf_image_file: str = None,
        flux_sample="pareto",
    ):
        super().__init__()
        assert n_bands == 1, "Only 1 band is supported"

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.n_bands = 1
        self.slen = slen
        self.background = torch.zeros((self.n_bands, self.slen, self.slen), dtype=torch.float32)
        self.background[...] = background
        self.noise_factor = noise_factor
        self.pixel_scale = pixel_scale

        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = 0.5

        self.min_a_d = min_a_d
        self.max_a_d = max_a_d
        self.min_a_b = min_a_b
        self.max_a_b = max_a_b

        self.flux_sample = flux_sample

        self.psf = load_psf_from_file(psf_image_file, self.pixel_scale)

    @staticmethod
    def _uniform(a, b):
        # uses pytorch to return a single float ~ U(a, b)
        unif = (a - b) * torch.rand(1) + b
        return unif.item()

    def _draw_pareto_flux(self):
        # return draw_pareto_maxed((1,), self.device, self.min_flux, self.max_flux, self.alpha)
        return draw_pareto_maxed((1,), "cpu", self.min_flux, self.max_flux, self.alpha)

    def __getitem__(self, idx):

        # create galaxy as mixture of Exponential + DeVacauleurs
        if self.flux_sample == "uniform":
            total_flux = self._uniform(self.min_flux, self.max_flux)
        elif self.flux_sample == "pareto":
            total_flux = self._draw_pareto_flux()
        else:
            raise NotImplementedError()

        disk_frac = self._uniform(0, 1)
        bulge_frac = 1 - disk_frac

        disk_flux = total_flux * disk_frac
        bulge_flux = total_flux * bulge_frac

        beta_radians = self._uniform(0, 2 * np.pi)

        components = []
        if disk_flux > 0:
            disk_q = self._uniform(0, 1)
            a_d = self._uniform(self.min_a_d, self.max_a_d)
            b_d = a_d * disk_q
            disk_hlr_arcsecs = np.sqrt(a_d * b_d)
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
                q=disk_q,
                beta=beta_radians * galsim.radians,
            )
            components.append(disk)

        if bulge_flux > 0:
            bulge_q = self._uniform(0, 1)
            a_b = self._uniform(self.min_a_b, self.max_a_b)
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
        image = torch.from_numpy(image.array).reshape(1, self.slen, self.slen)
        noiseless = image.clone()

        # add noise and background.
        image += self.background.mean()
        noise = image.sqrt() * torch.randn(*image.shape) * self.noise_factor
        image += noise

        return {"images": image, "background": self.background, "noiseless": noiseless}

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)


def draw_pareto_maxed(shape, device, f_min, f_max, alpha):
    # draw pareto conditioned on being less than f_max
    u_max = pareto_cdf(f_max, f_min, alpha)
    uniform_samples = torch.rand(*shape, device=device) * u_max
    return f_min / (1.0 - uniform_samples) ** (1 / alpha)


def pareto_cdf(x, f_min, alpha):
    return 1 - (f_min / x) ** alpha


def get_galaxy_latents(latents_file: str, n_latent_batches: int, autoencoder_ckpt: str = None):
    assert latents_file is not None
    latents_file = Path(latents_file)
    if latents_file.exists():
        latents = torch.load(latents_file, "cpu")
    else:
        vae = OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
        psf_image_file = latents_file.parent / "psField-000094-1-0012-PSF-image.npy"
        dataset = SDSSGalaxies(psf_image_file=psf_image_file)
        dataloader = dataset.train_dataloader()
        vae = vae.cuda()
        print("INFO: Creating latents from Galsim galaxies...")
        latents = generate_latents(vae, dataloader, n_latent_batches)
        torch.save(latents, latents_file)
        print(f"INFO: Saved latents to {latents_file}")
    return latents


def generate_latents(vae, dataloader, n_batches):
    """Induces a latent distribution for a non-probabilistic autoencoder."""
    latent_list = []
    enc = vae.get_encoder()
    with torch.no_grad():
        for _ in tqdm(range(n_batches)):
            galaxy = next(iter(dataloader))
            noiseless = galaxy["noiseless"].to(vae.device)
            latent_batch, _ = enc(noiseless, 0.0)
            latent_list.append(latent_batch)
    return torch.cat(latent_list, dim=0)
