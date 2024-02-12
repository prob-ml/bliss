from typing import Optional

import galsim
import numpy as np
import torch
from torch import Tensor

from bliss.datasets.lsst import PIXEL_SCALE, convert_mag_to_flux


def sample_stars(
    all_star_mags: np.ndarray,  # from star dc2 catalog
    slen: int,
    star_density: float,
    max_n_stars: int,
    max_shift: float = 0.5,  # between (0, 0.5)
):

    # counts
    exp_count = (slen * PIXEL_SCALE / 60) ** 2 * star_density
    n_stars = torch.tensor(_sample_poisson_n_sources(exp_count, max_n_stars))

    # locs
    locs = torch.zeros(max_n_stars, 2)
    locs[:n_stars, 0] = _uniform(-max_shift, max_shift, n_stars) + 0.5
    locs[:n_stars, 1] = _uniform(-max_shift, max_shift, n_stars) + 0.5

    mags = torch.zeros((max_n_stars,))
    fluxes = torch.zeros((max_n_stars,))
    star_log_fluxes = torch.zeros((max_n_stars,))

    mags[:n_stars] = np.random.choice(all_star_mags, size=(n_stars,), replace=True)  # noqa: WPS362
    fluxes[:n_stars] = convert_mag_to_flux(mags[:n_stars])  # noqa: WPS362
    star_log_fluxes[:n_stars] = torch.log(fluxes[:n_stars])  # noqa: WPS362

    return {
        "n_stars": n_stars,
        "locs": locs,
        "star_fluxes": fluxes,
        "star_log_fluxes": star_log_fluxes,
    }


def _render_star(
    flux: float, size: int, psf: galsim.GSObject, offset: Optional[Tensor] = None
) -> Tensor:
    """Draw a single star with Galsim."""
    assert offset is None or offset.shape == (2,)
    star = psf.withFlux(flux)  # creates a copy
    offset = offset if offset is None else offset.numpy()
    image = star.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset)
    return torch.from_numpy(image.array).reshape(1, size, size)


def render_stars(
    star_mags: Tensor,
    star_plocs: Tensor,
    slen: float,
    bp: float,
    psf: galsim.GSObject,
    max_n_stars: int,
):
    # single, constant PSF
    # single-band for `star_mags`
    assert star_mags.ndim == star_plocs.ndim == 2
    assert star_mags.shape[0] == star_plocs.shape[0]  # first dimension is max_n_stars
    assert star_plocs.shape[1] == 2

    size = slen + 2 * bp
    image = torch.zeros((1, size, size))  # single-band only
    noiseless_centered = torch.zeros((max_n_stars, 1, size, size))
    noiseless_uncentered = torch.zeros((max_n_stars, 1, size, size))

    for ii, (mag, ploc) in enumerate(zip(star_mags, star_plocs)):
        mag = mag.item()
        offset_x = ploc[1] + bp - size / 2
        offset_y = ploc[0] + bp - size / 2
        offset = torch.tensor([offset_x, offset_y])
        flux = convert_mag_to_flux(mag)
        star_uncentered = _render_star(flux, size, psf, offset=offset)
        star_centered = _render_star(flux, size, psf)
        noiseless_uncentered[ii] = star_uncentered
        noiseless_centered[ii] = star_centered
        image += star_uncentered

    return image, noiseless_centered, noiseless_uncentered


def _sample_poisson_n_sources(mean_sources, max_n_sources) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def _uniform(a, b, n_samples=1) -> Tensor:
    # uses pytorch to return a single float ~ U(a, b)
    return (a - b) * torch.rand(n_samples) + b
