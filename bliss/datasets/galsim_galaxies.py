import numpy as np
from pathlib import Path
from astropy.table import Table
from omegaconf import DictConfig
import galsim

from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import pytorch_lightning as pl


n_bands_dict = {1: ("r",), 6: ("y", "z", "i", "r", "g", "u")}


class GalsimGalaxies(pl.LightningDataModule, Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # assume 1 band everytime.
        self.cfg = cfg
        self.num_workers = cfg.dataset.num_workers
        self.batch_size = cfg.dataset.batch_size
        self.n_batches = cfg.dataset.n_batches

        params = self.cfg.dataset.params
        self.slen = int(params.slen)
        assert self.slen % 2 == 1, "Need divisibility by 2"
        self.n_bands = 1
        self.background = np.zeros(
            (self.n_bands, self.slen, self.slen), dtype=np.float32
        )
        self.background[...] = params.background
        self.pixel_scale = params.pixel_scale
        self.snr = params.snr

        # for adding noise to galsim images.
        self.rng = galsim.BaseDeviate(seed=999999)

        # small dummy psf
        self.psf = galsim.Gaussian(half_light_radius=0.2).withFlux(1.0)

    def __getitem__(self, idx):
        flux_avg = np.random.uniform(100, 1000)
        hlr = np.random.uniform(0.4, 1.0)  # arcseconds
        flux = (hlr / self.pixel_scale) ** 2 * np.pi * flux_avg  # approx

        # sample ellipticity
        l = np.random.uniform(0, 0.5)
        theta = np.random.uniform(0, 2 * np.pi)
        g1 = l * np.cos(theta)
        g2 = l * np.sin(theta)
        galaxy = (
            galsim.Gaussian(half_light_radius=hlr).shear(g1=g1, g2=g2).withFlux(flux)
        )
        gal_conv = galsim.Convolution(galaxy, self.psf)
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale
        )

        # add noise.
        poisson_noise = galsim.PoissonNoise(self.rng, sky_level=self.background.mean())
        image.addNoiseSNR(poisson_noise, snr=self.snr, preserve_flux=True)

        # add background
        image = image.array.reshape(1, self.slen, self.slen).astype(np.float32)
        image += self.background

        return {"images": image, "background": self.background}

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers
        )
