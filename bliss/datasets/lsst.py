from typing import Optional

import galcheat
import galsim
import numpy as np
import torch
from astropy import units as u
from galcheat.utilities import mag2counts
from torch import Tensor

PIXEL_SCALE = 0.2
MAX_MAG_GAL = 27.3
MAX_MAG_STAR = 26.0  # see histogram of dc2 star catalog in i-band
MIN_MAG = 0.0


def convert_mag_to_flux(mag: Tensor) -> Tensor:
    """Assuming gain = 1 always."""
    return torch.from_numpy(mag2counts(mag.numpy(), "LSST", "i").to_value("electron"))


def convert_flux_to_mag(counts: Tensor) -> Tensor:
    i_band = galcheat.get_survey("LSST").get_filter("i")

    flux = counts.numpy() * u.electron / i_band.full_exposure_time
    mag = flux.to(u.mag(u.electron / u.s)) + i_band.zeropoint

    return torch.from_numpy(mag.value)


def get_default_lsst_psf(
    atmospheric_model: Optional[str] = "Kolmogorov",
) -> galsim.GSObject:
    """Returns a synthetic LSST-like PSF in the i-band with an atmospheric and optical component.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        atmospheric_model: type of atmospheric model. Current options:
            ['Kolmogorov', 'Moffat', 'None'].

    Returns:
        Galsim PSF model as a galsim.GSObject.
    """

    # get info from galcheat
    lsst = galcheat.get_survey("LSST")
    i_band = lsst.get_filter("i")
    mirror_diameter = lsst.mirror_diameter.to_value("m")
    effective_area = lsst.effective_area.to_value("m2")
    fwhm = i_band.psf_fwhm.to_value("arcsec")  # = 0.79, atmospheric component
    effective_wavelength = i_band.effective_wavelength.to_value("angstrom")

    # define atmospheric psf
    if atmospheric_model == "Kolmogorov":
        atmospheric_psf_model = galsim.Kolmogorov(fwhm=fwhm)
    elif atmospheric_model == "Moffat":
        atmospheric_psf_model = galsim.Moffat(2, fwhm=fwhm)
    elif atmospheric_model == "None":
        atmospheric_psf_model = None
    else:
        raise NotImplementedError(
            f"The atmospheric model request '{atmospheric_model}' is incorrect or not implemented."
        )

    mirror_area = np.pi * (0.5 * mirror_diameter) ** 2
    area_ratio = effective_area / mirror_area
    if area_ratio <= 0 or area_ratio > 1:
        raise RuntimeError("Incompatible effective-area and mirror-diameter values.")
    obscuration_fraction = np.sqrt(1 - area_ratio)
    lambda_over_diameter = 3600 * np.degrees(1e-10 * effective_wavelength / mirror_diameter)
    optical_psf_model = galsim.Airy(
        lam_over_diam=lambda_over_diameter, obscuration=obscuration_fraction
    )

    psf_model = galsim.Convolve(atmospheric_psf_model, optical_psf_model)
    return psf_model.withFlux(1.0)  # pylint: disable=no-value-for-parameter


def column_to_tensor(table, colname):
    dtypes = {
        np.dtype(">i2"): int,
        np.dtype(">i4"): int,
        np.dtype(">i8"): int,
        np.dtype("bool"): bool,
        np.dtype(">f4"): np.float32,
        np.dtype(">f8"): np.float32,
        np.dtype("float32"): np.float32,
        np.dtype("float64"): np.dtype("float64"),
    }
    x = np.array(table[colname])
    dtype = dtypes[x.dtype]
    x = x.astype(dtype)
    return torch.from_numpy(x)


def table_to_dict(table):
    d = {}
    for p in table.columns:
        d[p] = column_to_tensor(table, p)
    return d


def catsim_row_to_galaxy_params(table, max_n_sources):
    names = (
        "fluxnorm_bulge",
        "fluxnorm_disk",
        "fluxnorm_agn",
        "a_b",
        "a_d",
        "b_b",
        "b_d",
        "pa_bulge",
        "i_ab",
        "flux",
    )

    params = torch.zeros((max_n_sources, len(names)))

    for ii, col in enumerate(table):
        for jj, n in enumerate(names):
            params[ii, jj] = column_to_tensor(col, n)

    return params
