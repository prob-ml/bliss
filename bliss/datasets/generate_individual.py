import galsim
import torch
from astropy.table import Table
from tqdm import tqdm

from bliss.datasets.background import add_noise_and_background, get_constant_background
from bliss.datasets.lsst import get_default_lsst_background
from bliss.datasets.render_utils import render_one_galaxy, sample_galaxy_params


def generate_individual_dataset(
    n_samples: int, catsim_table: Table, psf: galsim.GSObject, slen: int = 53, replace: bool = True
):
    """Like the function below but it only generates individual galaxies, so much faster to run."""

    background = get_constant_background(get_default_lsst_background(), (n_samples, 1, slen, slen))
    params, ids = sample_galaxy_params(catsim_table, n_samples, n_samples, replace=replace)
    assert params.shape == (n_samples, 11)
    gals = torch.zeros((n_samples, 1, slen, slen))
    for ii in tqdm(range(n_samples)):
        gal = render_one_galaxy(params[ii], psf, slen, offset=None)
        gals[ii] = gal

    # add noise
    noisy = add_noise_and_background(gals, background)

    return {
        "images": noisy,
        "background": background,
        "noiseless": gals,
        "galaxy_params": params,
        "indices": ids,
    }
