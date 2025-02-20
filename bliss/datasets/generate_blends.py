import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack, rearrange, reduce
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog, collate
from bliss.datasets.lsst import GALAXY_DENSITY, PIXEL_SCALE, STAR_DENSITY
from bliss.datasets.padded_tiles import render_padded_image
from bliss.datasets.render_utils import (
    add_noise,
    render_one_galaxy,
    render_one_star,
    sample_bernoulli,
    sample_galaxy_params,
    sample_poisson_n_sources,
    sample_star_fluxes,
    sample_uniform,
)


def generate_dataset(
    n_samples: int,
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    psf: galsim.GSObject,
    max_n_sources: int,
    galaxy_density: float = GALAXY_DENSITY,  # counts / sq. arcmin
    star_density: float = STAR_DENSITY,  # counts / sq. arcmin
    slen: int = 50,
    bp: int = 24,
    max_shift: float = 0.5,  # within tile, 0.5 -> maximum
    add_galaxies_in_padding: bool = True,
) -> dict[str, Tensor]:
    images_list = []
    noiseless_images_list = []
    uncentered_sources_list = []
    centered_sources_list = []
    paddings_list = []
    params_list = []

    size = slen + 2 * bp

    # internal region
    mean_sources_in = (galaxy_density + star_density) * (slen * PIXEL_SCALE / 60) ** 2
    mean_sources_out = (
        (galaxy_density + star_density) * (size**2 - slen**2) * (PIXEL_SCALE / 60) ** 2
    )
    galaxy_prob = galaxy_density / (galaxy_density + star_density)

    for _ in tqdm(range(n_samples)):
        full_cat = sample_full_catalog(
            catsim_table,
            all_star_mags,
            mean_sources=mean_sources_in,
            max_n_sources=max_n_sources,
            galaxy_prob=galaxy_prob,
            slen=slen,
            max_shift=max_shift,
        )
        center_noiseless, uncentered_sources, centered_sources = render_full_catalog(
            full_cat, psf, slen, bp
        )

        if add_galaxies_in_padding:
            padding_noiseless = render_padded_image(
                catsim_table, all_star_mags, mean_sources_out, galaxy_prob, psf, slen, bp
            )
        else:
            padding_noiseless = torch.zeros_like(center_noiseless)

        noiseless = center_noiseless + padding_noiseless
        noisy = add_noise(noiseless)

        images_list.append(noisy)
        noiseless_images_list.append(noiseless)
        uncentered_sources_list.append(uncentered_sources)
        centered_sources_list.append(centered_sources)
        params_list.append(full_cat.to_dict())

        # separately keep padding since it's needed in the deblender loss function
        # for that same purpose we also add central stars
        sbool = rearrange(full_cat["star_bools"], "1 ms 1 -> ms 1 1 1")
        all_stars = reduce(uncentered_sources * sbool, "ms 1 h w -> 1 h w", "sum")
        padding_with_stars_noiseless = padding_noiseless + all_stars
        paddings_list.append(padding_with_stars_noiseless)

    images, _ = pack(images_list, "* c h w")
    noiseless, _ = pack(noiseless_images_list, "* c h w")
    centered_sources, _ = pack(centered_sources_list, "* n c h w")
    uncentered_sources, _ = pack(uncentered_sources_list, "* n c h w")
    paddings, _ = pack(paddings_list, "* c h w")
    paramss = collate(params_list)

    assert centered_sources.shape[:3] == (n_samples, max_n_sources, 1)
    assert uncentered_sources.shape[:3] == (n_samples, max_n_sources, 1)

    return {
        "images": images,
        "noiseless": noiseless,
        "uncentered_sources": uncentered_sources,
        "centered_sources": centered_sources,
        "paddings": paddings,
        **paramss,
    }


def parse_dataset(dataset: dict[str, Tensor], tile_slen: int = 5):
    """Parse dataset into a tuple of (images, TileCatalog)."""
    params = dataset.copy()  # make a copy to not change argument.
    images = params.pop("images")
    paddings = params.pop("paddings")
    return images, TileCatalog(tile_slen, params), paddings


def render_full_catalog(full_cat: FullCatalog, psf: galsim.GSObject, slen: int, bp: int):
    size = slen + 2 * bp
    full_plocs = full_cat.plocs
    b, max_n_sources, _ = full_plocs.shape
    assert b == 1, "Only one batch supported for now."

    image = torch.zeros(1, size, size)
    centered_noiseless = torch.zeros(max_n_sources, 1, size, size)
    uncentered_noiseless = torch.zeros(max_n_sources, 1, size, size)

    n_sources = int(full_cat.n_sources.item())
    galaxy_params = full_cat["galaxy_params"][0]
    star_fluxes = full_cat["star_fluxes"][0]
    galaxy_bools = full_cat["galaxy_bools"][0]
    star_bools = full_cat["star_bools"][0]
    plocs = full_plocs[0]
    for ii in range(n_sources):
        offset_x = plocs[ii][1] + bp - size / 2
        offset_y = plocs[ii][0] + bp - size / 2
        offset = torch.tensor([offset_x, offset_y])
        if galaxy_bools[ii] == 1:
            source_uncentered = render_one_galaxy(galaxy_params[ii], psf, size, offset)
            source_centered = render_one_galaxy(galaxy_params[ii], psf, size, offset=None)
        elif star_bools[ii] == 1:
            source_uncentered = render_one_star(psf, star_fluxes[ii][0].item(), size, offset)
            source_centered = render_one_star(psf, star_fluxes[ii][0].item(), size, offset=None)
        else:
            continue
        centered_noiseless[ii] = source_centered
        uncentered_noiseless[ii] = source_uncentered
        image += source_uncentered

    return image, uncentered_noiseless, centered_noiseless


def sample_full_catalog(
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    galaxy_prob: float,
    slen: int = 50,
    max_shift: float = 0.5,
):
    params = sample_source_params(
        catsim_table,
        all_star_mags,
        mean_sources=mean_sources,
        max_n_sources=max_n_sources,
        galaxy_prob=galaxy_prob,
        slen=slen,
        max_shift=max_shift,
    )

    for p, q in params.items():
        if p != "n_sources":
            params[p] = rearrange(q, "n d -> 1 n d")

    return FullCatalog(slen, slen, params)


def sample_source_params(
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    galaxy_prob: float,
    slen: int = 100,
    max_shift: float = 0.5,
) -> dict[str, Tensor]:
    """Returns source parameters corresponding to a single blend."""
    n_sources = sample_poisson_n_sources(mean_sources, max_n_sources)
    params, _ = sample_galaxy_params(catsim_table, n_sources, max_n_sources)
    assert params.shape == (max_n_sources, 11)

    star_fluxes = sample_star_fluxes(all_star_mags, n_sources, max_n_sources)

    galaxy_bools = torch.zeros(max_n_sources, 1)
    star_bools = torch.zeros(max_n_sources, 1)
    galaxy_bools[:n_sources, :] = sample_bernoulli(galaxy_prob, n_sources)[:, None]
    star_bools[:n_sources, :] = 1 - galaxy_bools[:n_sources, :]

    locs = torch.zeros(max_n_sources, 2)
    locs[:n_sources, 0] = sample_uniform(-max_shift, max_shift, n_sources) + 0.5
    locs[:n_sources, 1] = sample_uniform(-max_shift, max_shift, n_sources) + 0.5
    plocs = locs * slen

    return {
        "n_sources": torch.tensor([n_sources]),
        "plocs": plocs,
        "galaxy_bools": galaxy_bools,
        "star_bools": star_bools,
        "galaxy_params": params * galaxy_bools,
        "star_fluxes": star_fluxes * star_bools,
        "fluxes": params[:, -1, None] * galaxy_bools + star_fluxes * star_bools,
    }


def get_full_catalog_from_dataset(ds: dict, slen: int):
    out = {}
    _params = (*FullCatalog.allowed_params, "plocs", "n_sources")
    for k in _params:
        if k in ds:
            out[k] = ds[k]
    return FullCatalog(slen, slen, out)
