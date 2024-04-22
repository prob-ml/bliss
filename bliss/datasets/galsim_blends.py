from typing import Optional

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack, rearrange, reduce
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.background import add_noise_and_background, get_constant_background
from bliss.datasets.lsst import PIXEL_SCALE, convert_mag_to_flux, get_default_lsst_background
from bliss.datasets.table_utils import catsim_row_to_galaxy_params


class SavedGalsimBlends(Dataset):
    def __init__(
        self,
        dataset_file: str,
        epoch_size: int,
        slen: int = 40,
        tile_slen: int = 4,
        keep_padding: bool = False,
    ) -> None:
        super().__init__()
        ds: dict[str, Tensor] = torch.load(dataset_file)
        self.epoch_size = epoch_size

        self.images = ds.pop("images").float()  # needs to be a float for NN
        self.background = ds.pop("background").float()

        # don't need for training
        ds.pop("individuals")
        ds.pop("noiseless")

        # avoid large memory usage if we don't need padding.
        if not keep_padding:
            ds.pop("paddings")
            self.paddings = torch.tensor([0]).float()
        else:
            self.paddings = ds.pop("paddings").float()
        self.keep_padding = keep_padding

        full_catalog = FullCatalog(slen, slen, ds)
        tile_catalogs = full_catalog.to_tile_params(tile_slen, ignore_extra_sources=True)
        self.tile_params = tile_catalogs.to_dict()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "background": self.background[index],
            "paddings": self.paddings[index] if self.keep_padding else self.paddings,
            **tile_params_ii,
        }


class SavedIndividualGalaxies(Dataset):
    def __init__(self, dataset_file: str, epoch_size: int) -> None:
        super().__init__()
        ds: dict[str, Tensor] = torch.load(dataset_file)
        self.epoch_size = epoch_size

        self.images = ds.pop("images").float()  # needs to be a float for NN
        self.background = ds.pop("background").float()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        return {
            "images": self.images[index],
            "background": self.background[index],
        }


def generate_dataset(
    n_samples: int,
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    psf: galsim.GSObject,
    max_n_sources: int,
    galaxy_density: float = 185,  # counts / sq. arcmin
    star_density: float = 10,  # counts / sq. arcmin
    slen: int = 40,
    bp: int = 24,
    max_shift: float = 0.5,  # within tile, 0.5 -> maximum
    add_galaxies_in_padding: bool = True,
) -> dict[str, Tensor]:

    images_list = []
    noiseless_images_list = []
    individuals_list = []
    paddings_list = []
    params_list = []

    size = slen + 2 * bp

    background = get_constant_background(get_default_lsst_background(), (n_samples, 1, size, size))

    # internal region
    mean_sources_in = (galaxy_density + star_density) * (slen * PIXEL_SCALE / 60) ** 2
    mean_sources_out = (
        (galaxy_density + star_density) * (size**2 - slen**2) * (PIXEL_SCALE / 60) ** 2
    )
    galaxy_prob = galaxy_density / (galaxy_density + star_density)

    for ii in tqdm(range(n_samples)):
        full_cat = sample_full_catalog(
            catsim_table,
            all_star_mags,
            mean_sources=mean_sources_in,
            max_n_sources=max_n_sources,
            slen=slen,
            max_shift=max_shift,
            galaxy_prob=galaxy_prob,
        )
        center_noiseless, individual_noiseless = render_full_catalog(full_cat, psf, slen, bp)

        if add_galaxies_in_padding:
            padding_noiseless = _render_padded_image(
                catsim_table, all_star_mags, mean_sources_out, galaxy_prob, psf, slen, bp
            )
        else:
            padding_noiseless = torch.zeros_like(center_noiseless)

        noiseless = center_noiseless + padding_noiseless
        noisy = add_noise_and_background(noiseless, background[ii, None])

        images_list.append(noisy)
        noiseless_images_list.append(noiseless)
        individuals_list.append(individual_noiseless)
        params_list.append(full_cat.to_tensor_dict())

        # separately keep padding since it's needed in the deblender loss function
        # for that same purpose we also add central stars
        sbool = rearrange(full_cat["star_bools"], "1 ms 1 -> ms 1 1 1")
        all_stars = reduce(individual_noiseless * sbool, "ms 1 h w -> 1 h w", "sum")
        padding_with_stars_noiseless = padding_noiseless + all_stars
        paddings_list.append(padding_with_stars_noiseless)

    images, _ = pack(images_list, "* c h w")
    noiseless_images, _ = pack(noiseless_images_list, "* c h w")
    individuals, _ = pack(individuals_list, "* n c h w")
    paddings, _ = pack(paddings_list, "* c h w")
    paramss = torch.cat(params_list, dim=0)

    assert individuals.shape[:3] == (n_samples, max_n_sources, 1)

    return {
        "images": images,
        "background": background,
        "noiseless": noiseless_images,
        "individuals": individuals,
        "paddings": paddings,
        **paramss,
    }


def _render_padded_image(
    catsim_table: Table,
    all_star_mags: np.ndarray,
    mean_sources: float,
    galaxy_prob: float,
    psf: galsim.GSObject,
    slen: int,
    bp: int,
):
    """We need to include galaxies outside the padding for realism. Return noiseless version."""
    size = slen + 2 * bp
    n_sources = _sample_poisson_n_sources(mean_sources, torch.inf)
    image = torch.zeros((1, size, size))

    # we don't need to record or keep track, just produce the image in padding
    # we will construct the image galaxy by galaxy
    for _ in range(n_sources):

        # offset always needs to be out of the central square
        x, y = _uniform_out_of_square(slen, size)
        offset = torch.tensor([x, y])

        is_galaxy = _bernoulli(galaxy_prob, 1).bool().item()
        if is_galaxy:
            params = _sample_galaxy_params(catsim_table, 1, 1)
            assert params.shape == (1, 10)
            one_galaxy_params = params[0]
            galaxy = _render_one_galaxy(one_galaxy_params, psf, size, offset)
            image += galaxy
        else:
            star_flux = _sample_star_fluxes(all_star_mags, 1, 1).item()
            star = _render_one_star(psf, star_flux, size, offset)
            image += star

    return image


def parse_dataset(dataset: dict[str, Tensor], tile_slen: int = 4):
    """Parse dataset into a tuple of (images, background, TileCatalog)."""
    params = dataset.copy()  # make a copy to not change argument.
    images = params.pop("images")
    background = params.pop("background")
    paddings = params.pop("paddings")
    return images, background, TileCatalog(tile_slen, params), paddings


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

    star_fluxes = _sample_star_fluxes(all_star_mags, n_sources, max_n_sources)

    galaxy_bools = torch.zeros(max_n_sources, 1)
    star_bools = torch.zeros(max_n_sources, 1)
    galaxy_bools[:n_sources, :] = _bernoulli(galaxy_prob, n_sources)[:, None]
    star_bools[:n_sources, :] = 1 - galaxy_bools[:n_sources, :]

    locs = torch.zeros(max_n_sources, 2)
    locs[:n_sources, 0] = _uniform(-max_shift, max_shift, n_sources) + 0.5
    locs[:n_sources, 1] = _uniform(-max_shift, max_shift, n_sources) + 0.5
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


def _sample_star_fluxes(all_star_mags: np.ndarray, n_sources: int, max_n_sources: int):
    star_fluxes = torch.zeros((max_n_sources, 1))
    star_mags = np.random.choice(all_star_mags, size=(n_sources,), replace=True)
    star_fluxes[:n_sources, 0] = convert_mag_to_flux(torch.from_numpy(star_mags))
    return star_fluxes


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
    individual_noiseless = torch.zeros(max_n_sources, 1, size, size)

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
            source = _render_one_galaxy(galaxy_params[ii], psf, size, offset)
        elif star_bools[ii] == 1:
            source = _render_one_star(psf, star_fluxes[ii][0].item(), size, offset)
        else:
            continue
        individual_noiseless[ii] = source
        image += source

    return image, individual_noiseless


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
    assert galaxy_params.device == torch.device("cpu") and galaxy_params.shape == (10,)
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


def _sample_poisson_n_sources(mean_sources: float, max_n_sources: int | float) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def _uniform(a, b, n_samples=1) -> Tensor:
    """Uses pytorch to return a single float ~ U(a, b)."""
    return (a - b) * torch.rand(n_samples) + b


def _uniform_out_of_square(a: float, b: float) -> float:
    """Returns two uniformly random numbers outside of central square with side-length a."""
    assert a < b
    x = _uniform(-b / 2, b / 2).item()
    if abs(x) < a / 2:
        is_left: bool = np.random.choice([False, True])
        if is_left:
            y = _uniform(-b / 2, -a / 2).item()
        else:
            y = _uniform(a / 2, b / 2).item()
    else:
        y = _uniform(-b / 2, b / 2).item()

    return x, y


def _bernoulli(prob, n_samples=1) -> Tensor:
    prob_list = [float(prob) for _ in range(n_samples)]
    return torch.bernoulli(torch.tensor(prob_list))
