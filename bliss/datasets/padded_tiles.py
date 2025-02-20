"""Functions to generate dataset of padded tiles directly with GalSim."""

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack, rearrange
from tqdm import tqdm

from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    PIXEL_SCALE,
    STAR_DENSITY,
)
from bliss.datasets.render_utils import (
    add_noise,
    render_one_galaxy,
    render_one_star,
    sample_bernoulli,
    sample_galaxy_params,
    sample_poisson_n_sources,
    sample_star_fluxes,
    sample_uniform,
    uniform_out_of_square,
)


def render_padded_image(
    catsim_table: Table,
    all_star_mags: np.ndarray,
    mean_sources: float,
    galaxy_prob: float,
    psf: galsim.GSObject,
    slen: int,
    bp: int,
):
    """Return noiseless image of galaxies only in padding (centroid outside `slen`)."""
    size = slen + 2 * bp
    n_sources = sample_poisson_n_sources(mean_sources, torch.inf)
    image = torch.zeros((1, size, size))

    # we don't need to record or keep track, just produce the image in padding
    # we will construct the image galaxy by galaxy
    for _ in range(n_sources):
        # offset always needs to be out of the central square
        x, y = uniform_out_of_square(slen, size)
        offset = torch.tensor([x, y])

        is_galaxy = sample_bernoulli(galaxy_prob, 1).bool().item()
        if is_galaxy:
            params, _ = sample_galaxy_params(catsim_table, 1, 1)
            assert params.shape == (1, 11)
            one_galaxy_params = params[0]
            galaxy = render_one_galaxy(one_galaxy_params, psf, size, offset)
            image += galaxy
        else:
            star_flux = sample_star_fluxes(all_star_mags, 1, 1).item()
            star = render_one_star(psf, star_flux, size, offset)
            image += star

    return image


def generate_padded_tiles(
    n_samples: int,
    catsim_table: Table,
    all_star_mags: np.ndarray,
    psf: galsim.GSObject,
    slen: int = 5,
    bp: int = 25,
    max_shift: float = 0.5,
    p_source_in: float | None = None,
    galaxy_prob: float | None = None,
):
    """Generated padded tiles with sources in the padding for training of ML models.

    At most 1 source is allowed in each padded tile for simplicity.
    """
    size = slen + bp * 2

    ptiles = []
    paddings = []
    uncentered_ns = []
    centered_ns = []
    all_n_sources = []
    all_locs = []
    all_galaxy_params = []
    all_galaxy_bools = []
    all_star_fluxes = []

    density = GALAXY_DENSITY + STAR_DENSITY

    if galaxy_prob is None:
        galaxy_prob = GALAXY_DENSITY / density

    for _ in tqdm(range(n_samples)):
        if p_source_in is None:
            mean_sources_in = density * (slen * PIXEL_SCALE / 60) ** 2
            n_sources = sample_poisson_n_sources(mean_sources_in, 1)
        else:
            n_sources = int(torch.distributions.Bernoulli(p_source_in).sample([1]).item())

        mean_sources_out = density * (size**2 - slen**2) * (PIXEL_SCALE / 60) ** 2

        locs = torch.zeros((2,))
        locs[0] = sample_uniform(-max_shift, max_shift, 1).item() + 0.5
        locs[1] = sample_uniform(-max_shift, max_shift, 1).item() + 0.5
        ploc = locs * slen

        offset_x = ploc[1] + bp - size / 2
        offset_y = ploc[0] + bp - size / 2
        offset = torch.tensor([offset_x, offset_y])

        galaxy_bool = torch.distributions.Bernoulli(galaxy_prob).sample([1]).item()

        if galaxy_bool == 1 and n_sources == 1:
            galaxy_params, _ = sample_galaxy_params(catsim_table, 1, 1)
            galaxy_params = rearrange(galaxy_params, "1 p -> p")
            star_flux = 0.0
            assert galaxy_params.shape == (11,)
            uncentered_noiseless = render_one_galaxy(galaxy_params, psf, size, offset=offset)
            centered_noiseless = render_one_galaxy(galaxy_params, psf, size, offset=None)
        elif galaxy_bool == 0 and n_sources == 1:
            star_flux = sample_star_fluxes(all_star_mags, 1, 1).item()
            galaxy_params = torch.zeros((11,)).float()
            uncentered_noiseless = render_one_star(psf, star_flux, size, offset=offset)
            centered_noiseless = render_one_star(psf, star_flux, size, offset=None)
        else:
            star_flux = 0.0
            galaxy_params = torch.zeros((11,)).float()
            uncentered_noiseless = torch.zeros((1, size, size)).float()
            centered_noiseless = torch.zeros((1, size, size)).float()

        padding = render_padded_image(
            catsim_table, all_star_mags, mean_sources_out, galaxy_prob, psf, slen, bp
        )
        noiseless = uncentered_noiseless + padding
        final_image = add_noise(noiseless)

        ptiles.append(final_image)
        paddings.append(padding)
        uncentered_ns.append(uncentered_noiseless)
        centered_ns.append(centered_noiseless)

        all_n_sources.append(torch.tensor(n_sources))
        all_locs.append(locs)
        all_galaxy_params.append(galaxy_params)
        all_galaxy_bools.append(torch.tensor(galaxy_bool).reshape(1))
        all_star_fluxes.append(torch.tensor(star_flux).reshape(1))

    images, _ = pack(ptiles, "* c h w")
    paddings, _ = pack(paddings, "* c h w")
    centered_sources, _ = pack(centered_ns, "* c h w")
    uncentered_sources, _ = pack(uncentered_ns, "* c h w")

    n_sources, _ = pack(all_n_sources, "*")
    locs, _ = pack(all_locs, "* xy")
    galaxy_params, _ = pack(all_galaxy_params, "* d")
    galaxy_bools, _ = pack(all_galaxy_bools, "* d")
    star_fluxes, _ = pack(all_star_fluxes, "* d")

    return {
        "images": images,
        "paddings": paddings,
        "uncentered_sources": uncentered_sources,
        "centered_sources": centered_sources,
        "tile_params": {
            "n_sources": n_sources,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "galaxy_bools": galaxy_bools,
            "star_fluxes": star_fluxes,
        },
    }
