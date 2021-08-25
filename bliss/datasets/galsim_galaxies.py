from collections import namedtuple

import galsim
import numpy as np
import pytorch_lightning as pl
import torch
from astropy import units as u
from astropy.table import Table
from speclite.filters import ab_reference_flux, load_filter
from torch.utils.data import DataLoader, Dataset

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
        "effective_area",  # [m^2]
        "pixel_scale",  # [arcsec/pixel]
        "airmass",
        "zeropoint_airmass",  # airmass at which zeropoint is calculated
        "filters",  # list of filters.
    ],
)

Filter = namedtuple(
    "Filter",
    [
        "band",
        "exp_time",  # [s]
        "extinction",
        "median_psf_fwhm",  # [arcsec]
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


def _setup_sdss_params(sdss_kwargs, psf_points):

    # directly from survey + filter.
    assert len(sdss_survey.filters) == 1
    assert sdss_survey.filters[0].band == "r"
    pixel_scale = sdss_survey.pixel_scale

    # setup sdss object and psf at a given point.
    assert len(list(sdss_kwargs["bands"])) == 1
    assert sdss_kwargs["bands"][0] == 2
    assert len(list(psf_points)) == 2
    sdss_data = SloanDigitalSkySurvey(**sdss_kwargs)
    local_psf = sdss_data.rcfgcs[0][-1]
    x, y = psf_points
    psf = local_psf.psf_at_points(0, x, y)
    psf_image = galsim.Image(psf, scale=pixel_scale)
    psf = galsim.InterpolatedImage(psf_image).withFlux(1.0)

    return pixel_scale, psf


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
        flux_sample="uniform",
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

        self.min_flux = min_flux
        self.max_flux = max_flux
        self.alpha = 0.5

        self.min_a_d = min_a_d
        self.max_a_d = max_a_d
        self.min_a_b = min_a_b
        self.max_a_b = max_a_b

        self.flux_sample = flux_sample

        self.pixel_scale, self.psf = _setup_sdss_params(sdss_kwargs, psf_points)

    @staticmethod
    def _uniform(a, b):
        # uses pytorch to return a single float ~ U(a, b)
        u = (a - b) * torch.rand(1) + b
        return u.item()

    def _draw_pareto_flux(self):
        # draw pareto conditioned on being less than f_max
        u_max = 1 - (self.min_flux / self.max_flux) ** self.alpha
        uniform_samples = torch.rand(1) * u_max
        return self.min_flux / (1.0 - uniform_samples) ** (1 / self.alpha)

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


class SDSSCatalogGalaxies(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        sdss_catalog: str,  # filepath
        sdss_kwargs: dict,
        num_workers=0,
        batch_size=32,
        bands=("r",),
        slen=53,
        background=865.0,
        noise_factor=0.05,
        psf_points=(450, 550),  # points in the SDSS frame
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.n_bands = len(bands)
        self.slen = slen
        self.background = torch.zeros((self.n_bands, self.slen, self.slen), dtype=torch.float32)
        self.background[...] = background
        self.noise_factor = noise_factor

        # directly from survey + filter.
        assert self.n_bands == 1
        assert bands[0] == 1
        self.pixel_scale, self.psf = _setup_sdss_params(sdss_kwargs, psf_points)

        # read sdss-formatted catalog table of entries.
        self.catalog = Table.read(sdss_catalog, format="ascii")

    @staticmethod
    def _get_sdss_galaxy(entry):
        components = []
        disk_flux = entry["expflux_r"]
        bulge_flux = entry["devflux_r"]

        if disk_flux > 0:
            disk_beta = np.radians(entry["expphi_r"])  # radians
            disk_hlr = entry["exprad_r"]  # arcsecs
            disk_q = entry["expab_r"]
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr).shear(
                q=disk_q,
                beta=disk_beta * galsim.radians,
            )
            components.append(disk)

        if bulge_flux > 0:
            bulge_beta = np.radians(entry["devphi_r"])
            bulge_hlr = entry["devrad_r"]
            bulge_q = entry["devab_r"]
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr).shear(
                q=bulge_q,
                beta=bulge_beta * galsim.radians,
            )
            components.append(bulge)

        return galsim.Add(components)

    def __getitem__(self, idx):
        entry = self.catalog[idx]
        galaxy = self._get_sdss_galaxy(entry)
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
        return len(self.catalog)

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
