import math
from typing import Dict

import btk
import pytorch_lightning as pl
import torch
from astropy.table import Table
from einops import rearrange
from galcheat.utilities import mag2counts, mean_sky_level
from galsim.gsobject import GSObject
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

from bliss.catalog import FullCatalog, get_is_on_from_n_sources
from bliss.datasets.background import ConstantBackground
from bliss.datasets.lsst import (
    PIXEL_SCALE,
    catsim_row_to_galaxy_params,
    column_to_tensor,
    convert_flux_to_mag,
)
from bliss.datasets.stars import render_stars, sample_stars
from bliss.reporting import get_single_galaxy_ellipticities


def _setup_blend_galaxy_generator(
    catalog_file: str,
    density: float,
    max_number: float,
    slen: int,
    bp: int,
    seed: int,
    max_mag: float = 27.3,
):
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_file)

    stamp_size = (slen + 2 * bp) * PIXEL_SCALE  # arcsecs

    sampling_function = btk.sampling_functions.DensitySampling(
        max_number=max_number,
        min_number=0,
        density=density,
        stamp_size=stamp_size,
        max_shift=slen * PIXEL_SCALE / 2,  # in arcseconds
        seed=seed,
        max_mag=max_mag,
        mag_name="i_ab",
    )

    survey = btk.survey.get_surveys("LSST")

    return btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        survey,
        batch_size=1,  # batching is taking care of by torch dataset
        stamp_size=stamp_size,
        njobs=1,
        add_noise="none",  # will add noise and background later
        seed=seed,  # use same seed here
    )


class GalsimBlends(Dataset):
    """Dataset of galsim blends."""

    def __init__(
        self,
        catalog_file: str,  # should point to 'OneSqDeg.fits'
        stars_file: str,  # should point to 'stars_med_june2018.fits'
        tile_slen: int,
        max_sources_per_tile: int,
        slen: int,
        bp: int,
        seed: int,  # for draw generator
        star_density: float = 10,  # counts / sq. arcmin
        galaxy_density: float = 185,  # counts / sq. arcmin
    ):
        super().__init__()
        self.seed = seed

        # images
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.slen = slen
        self.bp = bp
        self.size = self.slen + 2 * self.bp
        self.pixel_scale = PIXEL_SCALE

        # counts and densities
        self.galaxy_density = galaxy_density
        self.star_density = star_density

        # we just need something sensible here
        self.max_n_galaxies = math.ceil(galaxy_density * (self.slen * PIXEL_SCALE / 60) ** 2 * 2)
        self.max_n_stars = math.ceil(
            self.max_n_galaxies * (self.star_density / self.galaxy_density),
        )
        self.max_n_sources = self.max_n_galaxies + self.max_n_stars

        sky_level: float = mean_sky_level("LSST", "i").to_value("electron")
        self.background = ConstantBackground((sky_level,))

        # btk
        self.galaxy_generator = _setup_blend_galaxy_generator(
            catalog_file, self.galaxy_density, self.max_n_galaxies, self.slen, self.bp, self.seed
        )

        self.all_star_magnitudes = column_to_tensor(Table.read(stars_file), "i_ab")

    def sample(self):
        # sampling
        star_params = sample_stars(
            self.all_star_magnitudes, self.slen, self.star_density, self.max_n_stars
        )
        batch = next(self.galaxy_generator)
        blend_cat = batch.catalog_list[0]  # always only 1 batch
        psf = batch.psf[3]  # i-band index
        assert isinstance(psf, GSObject)

        # number of sources
        n_stars = star_params["n_stars"]
        n_galaxies = torch.tensor(len(blend_cat))
        n_sources = torch.tensor(n_stars.item() + n_galaxies.item())
        is_on_sources = get_is_on_from_n_sources(n_sources, self.max_n_sources)
        assert n_sources <= self.max_n_sources

        # locs
        # NOTE: BTK positions are w.r.t to top-left corner in pixels.
        star_locs = star_params["locs"]
        x, y = column_to_tensor(blend_cat, "x_peak"), column_to_tensor(blend_cat, "y_peak")
        locs_x = (x - self.bp) / self.slen
        locs_y = (y - self.bp) / self.slen
        galaxy_locs = torch.vstack((locs_y, locs_x)).T.reshape(-1, 2)
        locs = torch.zeros((self.max_n_sources, 2))
        locs[:n_galaxies, :] = galaxy_locs
        locs[n_galaxies : n_galaxies + n_stars, :] = star_locs[:n_stars]

        # bools
        galaxy_bools = torch.zeros((self.max_n_sources, 1))
        galaxy_bools[:n_galaxies, :] = 1
        star_bools = (1 - galaxy_bools) * is_on_sources.reshape(-1, 1)

        # star fluxes and log_fluxes
        star_fluxes = star_params["star_fluxes"].reshape(-1, 1)
        star_log_fluxes = star_params["star_log_fluxes"].reshape(-1, 1)
        new_star_fluxes = torch.zeros((self.max_n_sources, 1))
        new_star_log_fluxes = torch.zeros((self.max_n_sources, 1))
        new_star_fluxes[n_galaxies : n_galaxies + n_stars, :] = star_fluxes[:n_stars, :]
        new_star_log_fluxes[n_galaxies : n_galaxies + n_stars, :] = star_log_fluxes[:n_stars, :]

        # galaxy params
        gal_flux = mag2counts(blend_cat["i_ab"], "LSST", "i").to_value("electron")
        blend_cat["flux"] = gal_flux.astype(float)
        # NOTE: this function requires all galaxies to be in the front to work
        galaxy_params = catsim_row_to_galaxy_params(blend_cat, self.max_n_sources)

        # images
        stars_image, stars_isolated = render_stars(
            n_stars, star_fluxes, star_locs, self.slen, self.bp, psf, self.max_n_stars
        )
        galaxies_image = torch.from_numpy(batch.blend_images[0, 3, None])
        noiseless = stars_image + galaxies_image

        galaxies_isolated = torch.from_numpy(batch.isolated_images[0, :, 3, None])
        isolated_images = torch.zeros((self.max_n_sources, 1, self.size, self.size))
        isolated_images[:n_galaxies] = galaxies_isolated[:n_galaxies]  # noqa:WPS362
        isolated_images[n_galaxies : n_galaxies + n_stars] = stars_isolated[:n_stars]  # noqa:WPS362
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        image = _add_noise_and_background(noiseless, background)

        params_dict = {
            "n_sources": n_sources,
            "plocs": locs * self.slen,
            "galaxy_bools": galaxy_bools,
            "star_bools": star_bools,
            "star_fluxes": new_star_fluxes,
            "star_log_fluxes": new_star_log_fluxes,
            "galaxy_params": galaxy_params,
        }
        params_dict = {k: v.unsqueeze(0) for k, v in params_dict.items()}
        cat = FullCatalog(self.slen, self.slen, params_dict)
        return cat, image, noiseless, isolated_images, background, psf  # noqa: WPS227

    def _add_metrics(
        self,
        full_cat: FullCatalog,
        noiseless: Tensor,
        isolated_sources: Tensor,
        background: Tensor,
        psf: GSObject,
    ):
        scale = self.pixel_scale
        n_sources = int(full_cat.n_sources.item())
        galaxy_params = full_cat["galaxy_params"]
        galaxy_bools = full_cat["galaxy_bools"]

        # add important metrics to full catalog
        psf_numpy = psf.drawImage(nx=self.size, ny=self.size, scale=scale).array
        psf_tensor = torch.from_numpy(psf_numpy)

        single_sources = isolated_sources[:n_sources]
        single_sources = rearrange(single_sources, "n 1 h w -> n h w", n=n_sources)
        ellips = torch.zeros(self.max_n_sources, 2)
        e12 = get_single_galaxy_ellipticities(single_sources, psf_tensor, scale)
        ellips[:n_sources, :] = e12
        ellips = rearrange(ellips, "n g -> 1 n g", n=self.max_n_sources, g=2)
        ellips *= galaxy_bools  # star ellipticity measurements get zeroed out by definition

        # get snr and blendedness
        snr = torch.zeros(self.max_n_sources)
        blendedness = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            snr[ii] = _get_snr(isolated_sources[ii], background)
            blendedness[ii] = _get_blendedness(isolated_sources[ii], noiseless)
        snr = rearrange(snr, "n -> 1 n 1", n=self.max_n_sources)
        blendedness = rearrange(blendedness, "n -> 1 n 1", n=self.max_n_sources)

        # get magnitudes
        gal_fluxes = galaxy_params[0, :, -1]
        star_fluxes = full_cat["star_fluxes"][0, :, 0]
        mags = torch.zeros(self.max_n_sources)
        fluxes = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            gbool = galaxy_bools[0, ii, 0].item()
            flux = gal_fluxes[ii, None] if gbool == 1 else star_fluxes[ii, None]
            assert flux.item() > 0
            mag = convert_flux_to_mag(flux)
            fluxes[ii] = flux.item()
            mags[ii] = mag.item()
        mags = rearrange(mags, "n -> 1 n 1", n=self.max_n_sources)
        fluxes = rearrange(fluxes, "n -> 1 n 1", n=self.max_n_sources)

        # add to full catalog (needs to be in batches)
        full_cat["mags"] = mags
        full_cat["ellips"] = ellips
        full_cat["snr"] = snr
        full_cat["blendedness"] = blendedness
        full_cat["fluxes"] = fluxes
        return full_cat

    def _run_nan_check(self, *tensors):
        for t in tensors:
            assert not torch.any(torch.isnan(t))

    def __getitem__(self, idx):
        full_cat, images, noiseless, isolated_images, bg, psf = self.sample()
        full_cat = self._add_metrics(full_cat, noiseless, isolated_images, bg, psf)
        tile_cat = full_cat.to_tile_params(
            self.tile_slen, self.max_sources_per_tile, ignore_extra_sources=True
        )
        self._run_nan_check(images, bg, *tile_cat.to_dict().values(), *full_cat.to_dict().values())
        return TensorDict(
            {
                "images": images.unsqueeze(0),
                "background": bg.unsqueeze(0),
                "tile_params": tile_cat.to_dict(),
                "full_params": full_cat.to_dict(),
            },
            batch_size=[1],
        )


class SavedGalsimBlends(Dataset):

    def __init__(self, dataset_file: str, epoch_size: int) -> None:
        super().__init__()
        self.ds = torch.load(dataset_file)
        self.epoch_size = epoch_size
        assert len(self.ds["images"]) == self.epoch_size
        assert {"images", "background", "n_sources", "galaxy_bools"}.issubset(set(self.ds.keys()))
        assert {"star_log_fluxes", "locs"}.issubset(set(self.ds.keys()))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index) -> Dict[str, Tensor]:
        return {k: v[index] for k, v in self.ds.items()}


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _get_snr(image: Tensor, background: Tensor) -> float:
    image_with_background = image + background
    return torch.sqrt(torch.sum(image**2 / image_with_background)).item()


def _get_blendedness(single_source: Tensor, all_sources: Tensor) -> float:
    num = torch.sum(single_source * single_source).item()
    denom = torch.sum(single_source * all_sources).item()
    return 1 - num / denom
