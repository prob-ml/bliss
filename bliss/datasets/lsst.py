"""Utilites for calculating LSST survey related quantities."""

import galcheat
import galsim
import torch
from astropy import units as u  # noqa: WPS347
from btk.survey import get_surveys
from einops import rearrange
from galcheat.utilities import mag2counts, mean_sky_level
from torch import Tensor

PIXEL_SCALE = 0.2  # arcsecs / pixel
MAX_MAG_GAL = 27.3
MAX_MAG_STAR = 26.0  # see histogram of dc2 star catalog in i-band
MIN_MAG = 0


def convert_mag_to_flux(mag: Tensor) -> Tensor:
    """Assuming gain = 1 always."""
    return torch.from_numpy(mag2counts(mag.numpy(), "LSST", "i").to_value("electron"))


def convert_flux_to_mag(counts: Tensor) -> Tensor:
    i_band = galcheat.get_survey("LSST").get_filter("i")

    flux = counts.numpy() * u.electron / i_band.full_exposure_time  # pylint: disable=no-member
    mag = flux.to(u.mag(u.electron / u.s)) + i_band.zeropoint  # pylint: disable=no-member

    return torch.from_numpy(mag.value)


def get_default_lsst_psf() -> galsim.GSObject:
    """Returns a synthetic LSST-like PSF in the i-band with an atmospheric and optical component.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

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
    return mean_sky_level("LSST", "i").to_value("electron")
