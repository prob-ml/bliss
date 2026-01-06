"""Functions for creating a dataset of galaxies where there is a single galaxy in the central tile."""

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog, collate
from bliss.datasets.generate_blends import render_full_catalog
from bliss.datasets.lsst import GALAXY_DENSITY, PIXEL_SCALE
from bliss.datasets.padded_tiles import render_padded_image
from bliss.datasets.render_utils import (
    add_noise,
    sample_galaxy_params,
    sample_poisson_n_sources,
    uniform_out_of_square,
)


def _sample_source_params(
    catsim_table: Table,
    *,
    mean_sources: float,
    max_n_sources: int,
    slen: int,
    tile_slen: int,
) -> dict[str, Tensor]:
    """Returns source parameters corresponding to a single blend."""
    n_sources = sample_poisson_n_sources(mean_sources, max_n_sources)
    params, _ = sample_galaxy_params(
        catsim_table, n_galaxies=n_sources, max_n_sources=max_n_sources
    )
    assert params.shape == (max_n_sources, 11)

    galaxy_bools = torch.zeros(max_n_sources, 1)
    star_bools = torch.zeros(max_n_sources, 1)
    galaxy_bools[:n_sources, :] = 1.0

    _plocs = torch.zeros((max_n_sources, 2))
    for ii in range(n_sources):
        x, y = uniform_out_of_square(tile_slen, slen)
        _plocs[ii] = torch.tensor([y, x]) + slen / 2

    return {
        "n_sources": torch.tensor([n_sources]),
        "plocs": _plocs,
        "galaxy_bools": galaxy_bools,
        "galaxy_params": params * galaxy_bools,
        "star_bools": star_bools,
    }


def generate_central_sim_dataset(
    *,
    n_samples: int,
    catsim_table: Table,
    psf: galsim.GSObject,
    slen: int = 35,
    max_n_sources: int = 10,
    galaxy_density: float = GALAXY_DENSITY,  # counts / sq. arcmin
    tile_slen: int = 5,
    bp: int = 24,
    mag_cut_central: float = 25.3,
) -> dict[str, Tensor]:
    assert max_n_sources > 1, "max_n_sources must be greater than 1 for central simulation."
    size = slen + 2 * bp

    # regions density
    mean_sources_in = galaxy_density * (slen**2 - tile_slen**2) * (PIXEL_SCALE / 60) ** 2
    mean_sources_out = galaxy_density * (size**2 - slen**2) * (PIXEL_SCALE / 60) ** 2

    images = []
    noiseless_images = []
    uncentered_sources = []
    centered_sources = []
    paddings = []
    all_params = []

    mask_cat = catsim_table["i_ab"] < mag_cut_central
    bright_cat = catsim_table[mask_cat]
    for _ in tqdm(range(n_samples)):
        # sample parameters of central source (always 1 galaxy)
        plocs1 = torch.tensor([slen / 2, slen / 2]).view(1, 2)  # center of the tile
        params1, _ = sample_galaxy_params(bright_cat, n_galaxies=1, max_n_sources=1)
        assert params1.shape == (1, 11)
        assert plocs1.shape == (1, 2)

        # sample parameters of sources in central region, but outside central tile (still inferred)
        # combine full catalog with central source parameters
        params2_dict = _sample_source_params(
            catsim_table=catsim_table,
            mean_sources=mean_sources_in,
            max_n_sources=max_n_sources - 1,
            slen=slen,
            tile_slen=tile_slen,
        )

        # combine into full catalog with central source parameters
        n_sources = params2_dict["n_sources"].item() + 1
        params = torch.cat(
            [params1, params2_dict["galaxy_params"]],
        )
        plocs = torch.cat(
            [plocs1, params2_dict["plocs"]],
        )
        galaxy_bools = torch.cat(
            [torch.ones(1, 1), params2_dict["galaxy_bools"]],
        )
        assert plocs.shape == (max_n_sources, 2)
        assert params.shape == (max_n_sources, 11)
        assert galaxy_bools.shape == (max_n_sources, 1)
        full_cat = FullCatalog(
            slen,
            slen,
            {
                "n_sources": torch.tensor([n_sources]),
                "plocs": plocs.unsqueeze(0),
                "galaxy_bools": galaxy_bools.unsqueeze(0),
                "galaxy_params": params.unsqueeze(0),
                "star_bools": torch.zeros(1, max_n_sources, 1),
                "star_fluxes": torch.zeros(1, max_n_sources, 1),
            },
        )
        assert full_cat.n_sources.item() <= max_n_sources
        assert full_cat.n_sources.item() == full_cat["galaxy_bools"].sum().item()

        central_noiseless, uncentered_noiseless, centered_noiseless = render_full_catalog(
            full_cat, psf, slen, bp
        )

        # create padding region
        padding_noiseless = render_padded_image(
            catsim_table, np.array([np.nan]), mean_sources_out, 1.0, psf=psf, slen=slen, bp=bp
        )

        noiseless = central_noiseless + padding_noiseless
        image = add_noise(noiseless)

        images.append(image)
        noiseless_images.append(noiseless)
        paddings.append(padding_noiseless)
        uncentered_sources.append(uncentered_noiseless)
        centered_sources.append(centered_noiseless)
        all_params.append(full_cat.to_dict())

    images = pack(images, "* c h w")[0]
    noiseless_images = pack(noiseless_images, "* c h w")[0]
    paddings = pack(paddings, "* c h w")[0]
    uncentered_sources = pack(uncentered_sources, "* n c h w")[0]
    centered_sources = pack(centered_sources, "* n c h w")[0]
    params_all = collate(all_params)

    return {
        "images": images,
        "noiseless": noiseless_images,
        "paddings": paddings,
        "uncentered_sources": uncentered_sources,
        "centered_sources": centered_sources,
        **params_all,
    }
