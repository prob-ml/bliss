from typing import Dict, Optional, Tuple

import galsim
import numpy as np
import torch
from einops import rearrange
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

    # fluxes
    fluxes = torch.zeros((max_n_stars,))
    star_log_fluxes = torch.zeros((max_n_stars,))

    star_mags = np.random.choice(all_star_mags, size=(n_stars,), replace=True)
    star_mags = torch.from_numpy(star_mags)

    for ii in range(n_stars):
        fluxes[ii] = convert_mag_to_flux(star_mags[ii])
        star_log_fluxes[ii] = torch.log(fluxes[ii])

    return {
        "n_stars": n_stars,
        "locs": locs,
        "star_fluxes": rearrange(fluxes, "n -> n 1"),
        "star_log_fluxes": rearrange(star_log_fluxes, "n -> n 1"),
    }


def _render_star(
    flux: float, size: int, psf: galsim.GSObject, offset: Optional[Tensor] = None
) -> Tensor:
    """Draw a single star with Galsim."""
    assert offset is None or offset.shape == (2,)
    star = psf.withFlux(flux)  # creates a copy
    offset = offset if offset is None else offset.numpy()
    image = star.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset)
    image_tensor = torch.from_numpy(image.array)
    return rearrange(image_tensor, "h w -> 1 h w")


def render_stars(
    n_stars: int,
    star_fluxes: Tensor,
    star_locs: Tensor,
    slen: int,
    bp: int,
    psf: galsim.GSObject,
    max_n_stars: int,
) -> Tuple[Tensor, Tensor]:
    # single, constant PSF
    # single-band for `star_fluxes`
    assert star_fluxes.ndim == star_locs.ndim == 2
    assert star_fluxes.shape[0] == star_locs.shape[0]  # first dimension is max_n_stars
    assert star_locs.shape[1] == 2
    assert n_stars <= max_n_stars

    size = slen + 2 * bp
    image = torch.zeros((1, size, size))  # single-band only
    isolated_images = torch.zeros((max_n_stars, 1, size, size))

    for ii in range(n_stars):
        flux = star_fluxes[ii].item()
        ploc = star_locs[ii] * slen
        offset_x = ploc[1].item() + bp - size / 2
        offset_y = ploc[0].item() + bp - size / 2
        offset = torch.tensor([offset_x, offset_y])
        star_uncentered = _render_star(flux, size, psf, offset=offset)
        isolated_images[ii] = star_uncentered
        image += star_uncentered

    return image, isolated_images


def render_stars_from_params(
    star_params: Dict[str, Tensor], slen: int, bp: int, psf: galsim.GSObject, max_n_stars: int
):
    """Render stars but using directly the output for `sample_stars`."""
    n_stars = star_params["n_stars"]
    star_fluxes = star_params["star_fluxes"]
    locs = star_params["locs"]
    return render_stars(n_stars.item(), star_fluxes, locs, slen, bp, psf, max_n_stars)


def _sample_poisson_n_sources(mean_sources, max_n_sources) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def _uniform(a, b, n_samples=1) -> Tensor:
    # uses pytorch to return a single float ~ U(a, b)
    return (a - b) * torch.rand(n_samples) + b
