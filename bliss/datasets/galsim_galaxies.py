from collections import namedtuple
from speclite.filters import load_filter, ab_reference_flux
import numpy as np
import galsim
import torch

import astropy.units as u
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from bliss.datasets.sdss import SloanDigitalSkySurvey


def get_background(sky_brightness, filt, survey, B0=24):
    return get_flux(sky_brightness, filt, survey, B0=B0) * survey.pixel_scale ** 2


def get_flux(ab_mag, filt, survey, B0=24):
    """Convert source magnitude to flux.
    The calculation includes the effects of atmospheric extinction.
    Args:
        ab_magnitude(float): AB magnitude of source.
    Returns:
        float: Flux in detected electrons.
    """
    zeropoint = filt.zeropoint * survey.effective_area  # [s^-1]
    ab_mag += filt.extinction * (survey.airmass - survey.zeropoint_airmass)
    return filt.exp_time * zeropoint * 10 ** (-0.4 * (ab_mag - B0))


def calculate_zero_point(band_name="sdss2010-r", B0=24):
    # Credit: https://github.com/LSSTDESC/WeakLensingDeblending/issues/19
    filt = load_filter(band_name)
    return (
        (filt.convolve_with_function(ab_reference_flux) * 10 ** (-0.4 * B0))
        .to(1 / (u.s * u.m ** 2))  # pylint: disable=no-member
        .value
    )


Survey = namedtuple(
    "Survey",
    [
        "effective_area",  #  [m^2]
        "pixel_scale",  #  [arcsec/pixel]
        "airmass",
        "zeropoint_airmass",  # airmass at which zeropoint is calculated
        "filters",  # list of filters.
    ],
)

Filter = namedtuple(
    "Filter",
    [
        "band",
        "exp_time",  #  [s]
        "extinction",
        "median_psf_fwhm",  #  [arcsec]
        "effective_wavelength",  # [Angstroms]
        "limit_mag",
        "zeropoint",  # [s^-1 * m^-2]
        "sky_brightness",  # [mag]
    ],
)

# exp_time, pixel_scale, psf_fwhm, effective_wavelength, limit_mag from:
# https://www.sdss.org/dr12/scope/
# effective area from Gunn et al 2006 (https://arxiv.org/pdf/astro-ph/0602326.pdf, pg. 8)
# airmass from https://arxiv.org/pdf/1105.2320.pdf (table 1)
# extinction coefficent from https://iopscience.iop.org/article/10.1086/324741/pdf (table 24)
# zeropoint_airmass from https://speclite.readthedocs.io/en/latest/filters.html
# sky brighntess from: https://www.sdss.org/dr12/imaging/other_info/
# corresponds to a single exposure/filter.
sdss_survey = Survey(
    effective_area=3.58,
    pixel_scale=0.396,
    airmass=1.16,
    zeropoint_airmass=1.3,
    filters=[
        Filter(
            band="r",
            exp_time=53.9,
            extinction=0.1,
            median_psf_fwhm=1.3,
            effective_wavelength=6165,
            limit_mag=22.2,
            zeropoint=calculate_zero_point("sdss2010-r"),
            sky_brightness=21.06,  # stripe-82 specific, low surface brightness.
        )
    ],
)


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
        u = (a - b) * torch.rand(1) + b
        return u.item()

    def __getitem__(self, idx):
        flux_avg = self._uniform(self.min_flux, self.max_flux)
        hlr = self._uniform(self.min_hlr, self.max_hlr)  # arcseconds
        flux = (hlr / self.pixel_scale) ** 2 * np.pi * flux_avg  # approx

        # sample ellipticity
        l = self._uniform(0, self.max_e)
        theta = self._uniform(0, 2 * np.pi)
        g1 = l * np.cos(theta)
        g2 = l * np.sin(theta)
        galaxy = galsim.Gaussian(half_light_radius=hlr).shear(g1=g1, g2=g2).withFlux(flux)
        gal_conv = galsim.Convolution(galaxy, self.psf)
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale
        )

        # convert image to pytorch and reshape
        image = torch.from_numpy(image.array).reshape(1, self.slen, self.slen)

        # add noise and background.
        image += self.background
        noise = np.sqrt(image) * torch.randn(*image.shape) * self.noise_factor
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
        sdss_kwargs: dict,  # for obtaining PSF at points.
        num_workers=0,
        batch_size=64,
        n_batches=10,
        n_bands=1,
        slen=53,
        background=865.0,
        noise_factor=0.05,
        min_flux=1e3,
        max_flux=3.5e5,
        min_a_d=0.8,
        max_a_d=6.5,
        min_a_b=0.8,
        max_a_b=3.6,
        psf_points=(450, 550),  # points in the SDSS frame.
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

        # directly from survey + filter.
        assert len(sdss_survey.filters) == 1
        assert sdss_survey.filters[0].band == "r"
        self.survey = sdss_survey
        self.filt = self.survey.filters[0]
        self.pixel_scale = self.survey.pixel_scale

        self.min_flux = min_flux
        self.max_flux = max_flux
        self.min_a_d = min_a_d
        self.max_a_d = max_a_d
        self.min_a_b = min_a_b
        self.max_a_b = max_a_b

        # setup sdss object and psf at a given point.
        assert len(list(sdss_kwargs["bands"])) == 1
        assert sdss_kwargs["bands"][0] == 2
        assert len(list(psf_points)) == 2
        sdss_data = SloanDigitalSkySurvey(**sdss_kwargs)
        _psf = sdss_data.rcfgcs[0][-1]
        x, y = psf_points
        psf = _psf.psf_at_points(0, x, y)
        psf_image = galsim.Image(psf, scale=self.pixel_scale)
        self.psf = galsim.InterpolatedImage(psf_image).withFlux(1.0)

    def _uniform(self, a, b):
        # uses pytorch to return a single float ~ U(a, b)
        u = (a - b) * torch.rand(1) + b
        return u.item()

    def __getitem__(self, idx):

        # create galaxy as mixture of Exponential + DeVacauleurs
        total_flux = self._uniform(self.min_flux, self.max_flux)
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
                q=disk_q, beta=beta_radians * galsim.radians
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

        # add noise and background.
        image += self.background.mean()
        noise = image.sqrt() * torch.randn(*image.shape) * self.noise_factor
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


class SavedGalaxies(pl.LightningDataModule, Dataset):
    def __init__(self, data_file, batch_size=128):
        super().__init__()

        self.batch_size = batch_size

        self.data = torch.load(data_file)

        # Source Equation (4) in: https://arxiv.org/abs/2005.12039
        assert self.data["images"].shape[1] == 1, "Only 1 band supported"
        self.n_images = len(self.data["images"])

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image = self.data["images"][idx]
        background = self.data["background"]

        return {"images": image, "background": background}

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=0)