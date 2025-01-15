from typing import Optional

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import rearrange
from torch import Tensor

from bliss.datasets.lsst import BACKGROUND, PIXEL_SCALE, convert_mag_to_flux
from bliss.datasets.table_utils import catsim_row_to_galaxy_params


def add_noise(image: Tensor) -> Tensor:
    return image + BACKGROUND.sqrt() * torch.randn_like(image)


def render_one_star(
    psf: galsim.GSObject, flux: float, size: int, offset: Optional[Tensor] = None
) -> Tensor:
    assert offset is None or offset.shape == (2,)
    star = psf.withFlux(flux)
    offset = offset if offset is None else offset.numpy()
    image = star.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset)
    return rearrange(torch.from_numpy(image.array), "h w -> 1 h w")


def render_one_galaxy(
    galaxy_params: Tensor, psf: galsim.GSObject, size: int, offset: Optional[Tensor] = None
) -> Tensor:
    assert offset is None or offset.shape == (2,)
    assert galaxy_params.device == torch.device("cpu") and galaxy_params.shape == (11,)
    fnb, fnd, fnagn, ab, ad, bb, bd, pab, pad, _, total_flux = galaxy_params.numpy()  # noqa:WPS236

    disk_flux = total_flux * fnd / (fnd + fnb + fnagn)
    bulge_flux = total_flux * fnb / (fnd + fnb + fnagn)

    components = []
    if disk_flux > 0:
        assert bd > 0 and ad > 0 and pad > 0
        disk_q = bd / ad
        disk_hlr_arcsecs = np.sqrt(ad * bd)
        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            q=disk_q,
            beta=pad * galsim.degrees,
        )
        components.append(disk)
    if bulge_flux > 0:
        assert bb > 0 and ab > 0 and pab > 0
        bulge_q = bb / ab
        bulge_hlr_arcsecs = np.sqrt(ab * bb)
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
            q=bulge_q,
            beta=pab * galsim.degrees,
        )
        components.append(bulge)
    galaxy = galsim.Add(components)
    gal_conv = galsim.Convolution(galaxy, psf)
    offset = offset if offset is None else offset.numpy()
    galaxy_image = gal_conv.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset).array
    return rearrange(torch.from_numpy(galaxy_image), "h w -> 1 h w")


def sample_star_fluxes(all_star_mags: np.ndarray, n_sources: int, max_n_sources: int):
    star_fluxes = torch.zeros((max_n_sources, 1))
    star_mags = np.random.choice(all_star_mags, size=(n_sources,), replace=True)
    star_fluxes[:n_sources, 0] = convert_mag_to_flux(torch.from_numpy(star_mags))
    return star_fluxes


def sample_galaxy_params(
    catsim_table: Table, n_galaxies: int, max_n_sources: int, replace: bool = True
) -> tuple[Tensor, Tensor]:
    indices = np.random.choice(np.arange(len(catsim_table)), size=(n_galaxies,), replace=replace)

    rows = catsim_table[indices]
    mags = torch.from_numpy(rows["i_ab"].value.astype(np.float32))  # byte order
    gal_flux = convert_mag_to_flux(mags)
    rows["flux"] = gal_flux.numpy().astype(np.float32)

    ids = torch.from_numpy(rows["galtileid"].value.astype(int))
    return catsim_row_to_galaxy_params(rows, max_n_sources), ids


def sample_poisson_n_sources(mean_sources: float, max_n_sources: int | float) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def sample_uniform(a, b, n_samples=1) -> Tensor:
    """Uses pytorch to return a single float ~ U(a, b)."""
    return (a - b) * torch.rand(n_samples) + b


def sample_bernoulli(prob, n_samples=1) -> Tensor:
    prob_list = [float(prob) for _ in range(n_samples)]
    return torch.bernoulli(torch.tensor(prob_list))


def uniform_out_of_square(a: float, b: float) -> float:
    """Returns two uniformly random numbers in between squares of size a and b."""
    assert a < b
    x = sample_uniform(-b / 2, b / 2).item()
    if abs(x) < a / 2:
        is_left: bool = np.random.choice([False, True])
        if is_left:
            y = sample_uniform(-b / 2, -a / 2).item()
        else:
            y = sample_uniform(a / 2, b / 2).item()
    else:
        y = sample_uniform(-b / 2, b / 2).item()

    return x, y
