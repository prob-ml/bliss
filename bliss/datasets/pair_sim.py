import galsim
import torch
from astropy.table import Table
from einops import pack, rearrange
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import collate
from bliss.datasets.render_utils import (
    add_noise,
    render_one_galaxy,
    sample_galaxy_params,
    uniform_out_of_square,
)


def generate_pair_dataset(
    n_samples: int,
    catsim_table: Table,
    psf: galsim.GSObject,
    slen: int = 25,
    bp: int = 24,
    no_bar: bool = False,
    tile_slen: int = 5,
    out_square: float = 25.0,
) -> dict[str, Tensor]:
    """Simulation of a pair of galaxies where one is always at the center of image (and tile)."""
    size = slen + 2 * bp

    images = []
    uncentered_sources_list = []
    centered_sources_list = []
    params_list = []
    offsets = []

    for _ in tqdm(range(n_samples), disable=no_bar):
        # brightest one always at the center of the tile
        params, _ = sample_galaxy_params(catsim_table, n_galaxies=2, max_n_sources=2)
        assert params.shape == (2, 11)
        if params[0, -1] < params[1, -1]:
            params1 = params[1]
            params2 = params[0]
        else:
            params1 = params[0]
            params2 = params[1]

        galaxy1 = render_one_galaxy(params1, psf, size, offset=None)
        galaxy1 = rearrange(galaxy1, "1 h w -> 1 1 h w")

        # render one other galaxy in the padding
        # always outside the central tile
        offset = torch.tensor(uniform_out_of_square(tile_slen, out_square)).float()
        galaxy2 = render_one_galaxy(params2, psf, size, offset=offset)
        galaxy2_centered = render_one_galaxy(params2, psf, size, offset=None)
        galaxy2 = rearrange(galaxy2, "1 h w -> 1 1 h w")
        galaxy2_centered = rearrange(galaxy2_centered, "1 h w -> 1 1 h w")

        image = add_noise(galaxy1 + galaxy2)

        plocs1 = torch.tensor([slen / 2, slen / 2])
        plocs2 = torch.tensor([offset[1] + slen / 2, offset[0] + slen / 2])
        plocs = torch.stack([plocs1, plocs2], dim=0)

        cat_dict = {
            "n_sources": torch.tensor([2]),
            "plocs": plocs.reshape(1, 2, 2),
            "galaxy_bools": torch.ones(1, 2, 1),
            "galaxy_params": torch.stack([params1, params2]).reshape(1, 2, 11),
        }

        centered_sources = torch.concatenate([galaxy1, galaxy2_centered], dim=0)
        uncentered_sources = torch.concatenate([galaxy1, galaxy2], dim=0)

        images.append(image)
        uncentered_sources_list.append(uncentered_sources)
        centered_sources_list.append(centered_sources)
        params_list.append(cat_dict)
        offsets.append(offset)

    images, _ = pack(images, "* c h w")
    centered_sources, _ = pack(centered_sources_list, "* n c h w")
    uncentered_sources, _ = pack(uncentered_sources_list, "* n c h w")
    offsets = torch.stack(offsets, dim=0)
    params = collate(params_list)
    return {
        "images": images,
        "centered_sources": centered_sources,
        "uncentered_sources": uncentered_sources,
        **params,
        "offsets": offsets,  # of other galaxy
    }
