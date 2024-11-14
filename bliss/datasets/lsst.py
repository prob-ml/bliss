"""Utilites for calculating LSST survey related quantities."""

import galcheat
import galsim
import numpy as np
import torch
from astropy import units as u  # noqa: WPS347
from astropy.table import Table
from btk.survey import get_surveys
from einops import rearrange
from galcheat.utilities import mag2counts, mean_sky_level
from torch import Tensor

from bliss import HOME_DIR
from bliss.datasets.table_utils import column_to_tensor

PIXEL_SCALE = 0.2  # arcsecs / pixel

# for references on these cutoffs see notebooks:
# - tests/individual_galaxies_dist.ipynb
# - tests/star-snr.ipynb
MAX_MAG = 27.0  # both galaxies and stars
MIN_STAR_MAG = 20.0  # stars with lower magnitude have > 1000 SNR

# see notebook `tests/back-of-the-envelope.ipynb`
GALAXY_DENSITY = 160  # arcmin^{-2}, with mag cut above
STAR_DENSITY = 15.5  # arcmin^{-2}


DATA_DIR = HOME_DIR / "data"


def convert_mag_to_flux(mag: Tensor) -> Tensor:
    """Assuming gain = 1 always."""
    return torch.from_numpy(mag2counts(mag.numpy(), "LSST", "i").to_value("electron")).float()


def convert_flux_to_mag(counts: Tensor) -> Tensor:
    i_band = galcheat.get_survey("LSST").get_filter("i")

    flux = counts.numpy() * u.electron / i_band.full_exposure_time  # pylint: disable=no-member
    mag = flux.to(u.mag(u.electron / u.s)) + i_band.zeropoint  # pylint: disable=no-member

    return torch.from_numpy(mag.value)


def get_default_lsst_psf() -> galsim.GSObject:
    """Returns a synthetic LSST-like PSF in the i-band with an atmospheric and optical component.

    Returns:
        Galsim PSF model as a galsim.GSObject.
    """
    lsst = get_surveys("LSST")
    i_band = lsst.get_filter("i")
    return i_band.psf


def get_default_lsst_psf_tensor(slen: int) -> Tensor:
    psf_obj = get_default_lsst_psf()
    psf_array = psf_obj.drawImage(nx=slen, ny=slen, scale=PIXEL_SCALE).array
    psf_tensor = torch.from_numpy(psf_array)
    return rearrange(psf_tensor, "h w -> 1 h w", h=slen, w=slen).float()


def get_default_lsst_background() -> float:
    return mean_sky_level("LSST", "i").to_value("electron").astype(np.float32).item()


BACKGROUND = torch.tensor(get_default_lsst_background())


def prepare_final_galaxy_catalog() -> Table:
    """Function to globally apply cuts to CATSIM catalog for all datasets."""
    cat = Table.read(DATA_DIR / "OneDegSq.fits")
    mask = cat["i_ab"] < MAX_MAG
    return cat[mask]


def prepare_final_star_catalog() -> Tensor:
    table = Table.read(DATA_DIR / "stars_med_june2018.fits")
    mags = column_to_tensor(table, "i_ab")
    mask = torch.logical_and(mags > MIN_STAR_MAG, mags < MAX_MAG)
    return mags[mask]
