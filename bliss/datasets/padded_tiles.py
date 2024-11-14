"""Functions to generate dataset of padded tiles directly with GalSim."""

import galsim
import numpy as np
import torch
from astropy.table import Table

from bliss.datasets.render_utils import (
    render_one_galaxy,
    render_one_star,
    sample_bernoulli,
    sample_galaxy_params,
    sample_poisson_n_sources,
    sample_star_fluxes,
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
