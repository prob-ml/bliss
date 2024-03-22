from typing import Optional

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack, rearrange
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.background import add_noise_and_background, get_constant_background
from bliss.datasets.lsst import PIXEL_SCALE, convert_mag_to_flux, get_default_lsst_background
from bliss.datasets.table_utils import catsim_row_to_galaxy_params


class SavedGalsimBlends(Dataset):
    def __init__(
        self, dataset_file: str, epoch_size: int, slen: int = 40, tile_slen: int = 4
    ) -> None:
        super().__init__()
        self.ds: dict[str, Tensor] = torch.load(dataset_file)
        self.epoch_size = epoch_size

        self.images = self.ds.pop("images")
        self.background = self.ds.pop("background")

        full_catalog = FullCatalog(slen, slen, self.ds)
        tile_catalogs = full_catalog.to_tile_params(tile_slen, ignore_extra_sources=True)
        self.tile_params = tile_catalogs.to_dict()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "background": self.background[index],
            **tile_params_ii,
        }


def generate_dataset(
    n_samples: int,
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    psf: galsim.GSObject,
    slen: int = 40,
    bp: int = 24,
    max_shift: float = 0.5,
    galaxy_prob: float = 0.9,
) -> dict[str, Tensor]:

    images = []
    paramss = []

    size = slen + 2 * bp

    background = get_constant_background(get_default_lsst_background(), (n_samples, 1, size, size))
    for ii in tqdm(range(n_samples)):
        full_cat = sample_full_catalog(
            catsim_table, all_star_mags, mean_sources, max_n_sources, slen, max_shift, galaxy_prob
        )
        image, _, _ = render_full_catalog(full_cat, psf, slen, bp)
        noisy = add_noise_and_background(image, background[ii, None])
        images.append(noisy)
        paramss.append(full_cat.to_tensor_dict())

    images, _ = pack(images, "* c h w")
    paramss = torch.cat(paramss, dim=0)

    return {"images": images, "background": background, **paramss}


def sample_source_params(
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    slen: int = 40,
    max_shift: float = 0.5,
    galaxy_prob: float = 0.9,
) -> dict[str, Tensor]:
    """Returns a single batch of source parameters."""
    n_sources = _sample_poisson_n_sources(mean_sources, max_n_sources)
    params = _sample_galaxy_params(catsim_table, n_sources, max_n_sources)
    assert params.shape == (max_n_sources, 10)

    star_fluxes = torch.zeros((max_n_sources, 1))
    star_mags = np.random.choice(all_star_mags, size=(n_sources,), replace=True)
    star_fluxes[:n_sources, 0] = convert_mag_to_flux(torch.from_numpy(star_mags))

    locs = torch.zeros(max_n_sources, 2)
    locs[:n_sources, 0] = _uniform(-max_shift, max_shift, n_sources) + 0.5
    locs[:n_sources, 1] = _uniform(-max_shift, max_shift, n_sources) + 0.5
    plocs = locs * slen

    galaxy_bools = torch.zeros(max_n_sources, 1)
    galaxy_bools[:n_sources, :] = _bernoulli(galaxy_prob, n_sources)[:, None]
    star_bools = torch.zeros(max_n_sources, 1)
    star_bools[:n_sources, :] = 1 - galaxy_bools[:n_sources, :]

    return {
        "n_sources": torch.tensor([n_sources]),
        "plocs": plocs,
        "galaxy_bools": galaxy_bools,
        "star_bools": star_bools,
        "galaxy_params": params * galaxy_bools,
        "star_fluxes": star_fluxes * star_bools,
        "fluxes": params[:, -1, None] * galaxy_bools + star_fluxes * star_bools,
    }


def sample_full_catalog(
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    slen: int = 40,
    max_shift: float = 0.5,
    galaxy_prob: float = 0.9,
):
    params = sample_source_params(
        catsim_table, all_star_mags, mean_sources, max_n_sources, slen, max_shift, galaxy_prob
    )

    for p, q in params.items():
        if p != "n_sources":
            params[p] = rearrange(q, "n d -> 1 n d")

    return FullCatalog(slen, slen, params)


def render_full_catalog(full_cat: FullCatalog, psf: galsim.GSObject, slen: int, bp: int):
    size = slen + 2 * bp
    full_plocs = full_cat.plocs
    b, max_n_sources, _ = full_plocs.shape
    assert b == 1, "Only one batch supported for now."

    image = torch.zeros(1, size, size)
    noiseless_centered = torch.zeros(max_n_sources, 1, size, size)
    noiseless_uncentered = torch.zeros(max_n_sources, 1, size, size)

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
            centered = _render_one_galaxy(galaxy_params[ii], psf, size)
            uncentered = _render_one_galaxy(galaxy_params[ii], psf, size, offset)
        elif star_bools[ii] == 1:
            centered = _render_one_star(psf, star_fluxes[ii][0].item(), size)
            uncentered = _render_one_star(psf, star_fluxes[ii][0].item(), size, offset)
        else:
            continue
        noiseless_centered[ii] = centered
        noiseless_uncentered[ii] = uncentered
        image += uncentered

    return image, noiseless_centered, noiseless_uncentered


def _sample_galaxy_params(catsim_table: Table, n_galaxies: int, max_n_sources: int) -> Tensor:
    indices = np.random.choice(np.arange(len(catsim_table)), size=(n_galaxies,), replace=True)

    rows = catsim_table[indices]
    mags = torch.from_numpy(rows["i_ab"].value.astype(float))  # byte order
    gal_flux = convert_mag_to_flux(mags)
    rows["flux"] = gal_flux.numpy().astype(float)

    return catsim_row_to_galaxy_params(rows, max_n_sources)


def _render_one_star(
    psf: galsim.GSObject, flux: float, size: int, offset: Optional[Tensor] = None
) -> Tensor:
    assert offset is None or offset.shape == (2,)
    star = psf.withFlux(flux)
    offset = offset if offset is None else offset.numpy()
    image = star.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset)
    return rearrange(torch.from_numpy(image.array), "h w -> 1 h w")


def _render_one_galaxy(
    galaxy_params: Tensor, psf: galsim.GSObject, size: int, offset: Optional[Tensor] = None
) -> Tensor:
    assert offset is None or offset.shape == (2,)
    assert galaxy_params.device == torch.device("cpu")
    fnb, fnd, fnagn, ab, ad, bb, bd, pa, _, total_flux = galaxy_params.numpy()  # noqa:WPS236

    disk_flux = total_flux * fnd / (fnd + fnb + fnagn)
    bulge_flux = total_flux * fnb / (fnd + fnb + fnagn)

    components = []
    if disk_flux > 0:
        disk_q = bd / ad
        disk_hlr_arcsecs = np.sqrt(ad * bd)
        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            q=disk_q,
            beta=pa * galsim.radians,
        )
        components.append(disk)
    if bulge_flux > 0:
        bulge_q = bb / ab
        bulge_hlr_arcsecs = np.sqrt(ab * bb)
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
            q=bulge_q,
            beta=pa * galsim.radians,
        )
        components.append(bulge)
    galaxy = galsim.Add(components)
    gal_conv = galsim.Convolution(galaxy, psf)
    offset = offset if offset is None else offset.numpy()
    galaxy_image = gal_conv.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset).array
    return rearrange(torch.from_numpy(galaxy_image), "h w -> 1 h w")


def _sample_poisson_n_sources(mean_sources, max_n_sources) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def _uniform(a, b, n_samples=1) -> Tensor:
    # uses pytorch to return a single float ~ U(a, b)
    return (a - b) * torch.rand(n_samples) + b


def _bernoulli(prob, n_samples=1) -> Tensor:
    prob_list = [float(prob) for _ in range(n_samples)]
    return torch.bernoulli(torch.tensor(prob_list))
