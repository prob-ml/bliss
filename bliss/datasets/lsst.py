from typing import Optional

import galcheat
import galsim
import numpy as np
import torch
from torch import Tensor


def convert_mag_to_flux(mag: Tensor) -> Tensor:
    raise NotImplementedError


def convert_flux_to_mag(flux: Tensor) -> Tensor:
    raise NotImplementedError


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
    fwhm = i_band.psf_fwhm.to_value("arcsec")
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
