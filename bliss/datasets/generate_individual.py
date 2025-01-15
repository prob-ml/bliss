import galsim
import torch
from astropy.table import Table
from tqdm import tqdm

from bliss.datasets.render_utils import add_noise, render_one_galaxy, sample_galaxy_params


def generate_individual_dataset(
    n_samples: int, catsim_table: Table, psf: galsim.GSObject, slen: int = 53, replace: bool = True
):
    """Like the function below but it only generates individual galaxies, so much faster to run."""

    params, ids = sample_galaxy_params(catsim_table, n_samples, n_samples, replace=replace)
    assert params.shape == (n_samples, 11)
    gals = torch.zeros((n_samples, 1, slen, slen))
    for ii in tqdm(range(n_samples)):
        gal = render_one_galaxy(params[ii], psf, slen, offset=None)
        gals[ii] = gal

    noisy = add_noise(gals)

    return {
        "images": noisy,
        "noiseless": gals,
        "galaxy_params": params,
        "indices": ids,
    }
