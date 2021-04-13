from collections import namedtuple
from speclite.filters import load_filter, ab_reference_flux
import numpy as np
from omegaconf import DictConfig
import galsim
import torch

import astropy.units as u
from astropy.table import Table
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


def get_catsim_galaxy(entry, filt, survey, no_disk=False, no_bulge=False, no_agn=False):
    """Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)"""

    components = []
    total_flux = get_flux(entry[filt.band + "_ab"], filt, survey)
    # Calculate the flux of each component in detected electrons.
    total_fluxnorm = entry["fluxnorm_disk"] + entry["fluxnorm_bulge"] + entry["fluxnorm_agn"]
    disk_flux = 0.0 if no_disk else entry["fluxnorm_disk"] / total_fluxnorm * total_flux
    bulge_flux = 0.0 if no_bulge else entry["fluxnorm_bulge"] / total_fluxnorm * total_flux
    agn_flux = 0.0 if no_agn else entry["fluxnorm_agn"] / total_fluxnorm * total_flux

    if disk_flux + bulge_flux + agn_flux == 0:
        raise ValueError("Source not visible, check catalog values.")

    if disk_flux > 0:
        beta_radians = np.radians(entry["pa_disk"])
        if bulge_flux > 0:
            assert entry["pa_disk"] == entry["pa_bulge"], "Sersic components have different beta."
        a_d, b_d = entry["a_d"], entry["b_d"]
        disk_hlr_arcsecs = np.sqrt(a_d * b_d)
        disk_q = b_d / a_d
        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            q=disk_q, beta=beta_radians * galsim.radians
        )
        components.append(disk)

    if bulge_flux > 0:
        beta_radians = np.radians(entry["pa_bulge"])
        a_b, b_b = entry["a_b"], entry["b_b"]
        bulge_hlr_arcsecs = np.sqrt(a_b * b_b)
        bulge_q = b_b / a_b
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
            q=bulge_q, beta=beta_radians * galsim.radians
        )
        components.append(bulge)

    if agn_flux > 0:
        agn = galsim.Gaussian(flux=agn_flux, sigma=1e-8)
        components.append(agn)

    profile = galsim.Add(components)
    return profile


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


class SDSSGalaxies(pl.LightningDataModule, Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # assume 1 band everytime for now ('r' band).

        # general dataset parameters.
        self.cfg = cfg
        self.num_workers = cfg.dataset.num_workers
        self.batch_size = cfg.dataset.batch_size
        self.n_batches = cfg.dataset.n_batches

        # image paraemters.
        params = self.cfg.dataset.params
        self.slen = int(params.slen)
        self.n_bands = 1
        self.background = np.zeros((self.n_bands, self.slen, self.slen), dtype=np.float32)
        self.background[...] = params.background
        self.noise_factor = params.noise_factor

        # directly from survey + filter.
        assert len(sdss_survey.filters) == 1
        assert sdss_survey.filters[0].band == "r"
        self.survey = sdss_survey
        self.filt = self.survey.filters[0]
        self.pixel_scale = self.survey.pixel_scale

        # read cosmodc2 table of entries.
        self.catalog = Table.read(cfg.dataset.cosmoDC2_file)

        # setup sdss object and obtain psf at a given point.
        assert len(list(cfg.dataset.sdss.bands)) == 1
        assert cfg.dataset.sdss.bands[0] == 2
        assert len(list(cfg.dataset.psf.psf_points)) == 2
        sdss_data = SloanDigitalSkySurvey(**cfg.dataset.sdss)
        _psf = sdss_data.rcfgcs[0][-1]
        x, y = cfg.dataset.psf.psf_points
        psf = _psf.psf_at_points(0, x, y)
        psf_image = galsim.Image(psf, scale=self.pixel_scale)
        self.psf = galsim.InterpolatedImage(psf_image).withFlux(1.0)
        self.psf_fwhm = self.psf.calculateFWHM()  # arcsecs

    def __getitem__(self, idx):
        _idx = np.random.randint(len(self.catalog))
        entry = self.catalog[_idx]
        galaxy = get_catsim_galaxy(entry, self.filt, self.survey)
        gal_conv = galsim.Convolution(galaxy, self.psf)
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale
        )
        image = image.array.reshape(1, self.slen, self.slen).astype(np.float32)

        # add noise and background.
        image += self.background.mean()
        noise = np.sqrt(image) * np.random.randn(*image.shape) * self.noise_factor
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


class ToyGaussian(pl.LightningDataModule, Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # assume 1 band everytime.
        self.cfg = cfg
        self.num_workers = cfg.dataset.num_workers
        self.batch_size = cfg.dataset.batch_size
        self.n_batches = cfg.dataset.n_batches

        if self.num_workers > 0:
            raise NotImplementedError(
                "There is a problem where seed gets reset with multiple workers,"
                "resulting in same galaxy in every epoch."
            )

        params = self.cfg.dataset.params
        self.slen = int(params.slen)
        self.n_bands = 1
        self.background = np.zeros((self.n_bands, self.slen, self.slen), dtype=np.float32)
        self.background[...] = params.background
        self.pixel_scale = params.pixel_scale
        self.noise_factor = params.noise_factor

        # small dummy psf
        self.psf = galsim.Gaussian(fwhm=params.psf_fwhm).withFlux(1.0)
        self.min_flux = params.min_flux
        self.max_flux = params.max_flux
        self.min_hlr = params.min_hlr
        self.max_hlr = params.max_hlr
        self.max_e = params.max_e

    def __getitem__(self, idx):
        flux_avg = np.random.uniform(self.min_flux, self.max_flux)
        hlr = np.random.uniform(self.min_hlr, self.max_hlr)  # arcseconds
        flux = (hlr / self.pixel_scale) ** 2 * np.pi * flux_avg  # approx

        # sample ellipticity
        l = np.random.uniform(0, self.max_e)
        theta = np.random.uniform(0, 2 * np.pi)
        g1 = l * np.cos(theta)
        g2 = l * np.sin(theta)
        galaxy = galsim.Gaussian(half_light_radius=hlr).shear(g1=g1, g2=g2).withFlux(flux)
        gal_conv = galsim.Convolution(galaxy, self.psf)
        image = gal_conv.drawImage(
            nx=self.slen, ny=self.slen, method="auto", scale=self.pixel_scale
        )

        # add noise and background.
        image = image.array.reshape(1, self.slen, self.slen).astype(np.float32)
        image += self.background
        noise = np.sqrt(image) * np.random.randn(*image.shape) * self.noise_factor
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
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.batch_size = cfg.dataset.batch_size
        self.data = torch.load(cfg.dataset.data_file)

        # Source Equation (4) in: https://arxiv.org/abs/2005.12039
        assert self.data["images"].shape[1] == 1, "Only 1 band supported"
        self.do_norm = cfg.dataset.do_norm
        self.beta = 2.5
        self.norm = self.data["images"].reshape(len(self), 1, -1).max(axis=-1).values.mean()

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        _idx = np.random.randint(len(self))
        image = self.data["images"][_idx]
        background = self.data["background"]

        if self.do_norm:
            image = torch.tanh(torch.arcsinh(self.beta * image / self.norm))
            background = torch.tanh(torch.arcsinh(self.beta * background / self.norm))

        return {"images": image, "background": background}

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=0)
