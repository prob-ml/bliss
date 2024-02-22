import galcheat
import galsim
import numpy as np
import torch
from astropy import units as u
from btk.survey import get_surveys
from galcheat.utilities import mag2counts, mean_sky_level
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


def get_default_lsst_psf() -> galsim.GSObject:
    """Returns a synthetic LSST-like PSF in the i-band with an atmospheric and optical component.

    Credit: WeakLensingDeblending (https://github.com/LSSTDESC/WeakLensingDeblending)

    Args:
        atmospheric_model: type of atmospheric model. Current options:
            ['Kolmogorov', 'Moffat', 'None'].

    Returns:
        Galsim PSF model as a galsim.GSObject.
    """
    lsst = get_surveys("LSST")
    i_band = lsst.get_filter("i")
    return i_band.psf


def get_default_lsst_background() -> float:
    return mean_sky_level("LSST", "i").to_value("electron")


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
