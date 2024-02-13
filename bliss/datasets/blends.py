import math
from typing import Dict, List, Optional

import btk
import pytorch_lightning as pl
import torch
from astropy.table import Table
from einops import rearrange
from galcheat.utilities import mean_sky_level
from galsim.gsobject import GSObject
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bliss.catalog import FullCatalog, get_is_on_from_n_sources
from bliss.datasets.background import ConstantBackground
from bliss.datasets.galsim_stars import render_stars, sample_stars
from bliss.datasets.lsst import (
    PIXEL_SCALE,
    catsim_row_to_galaxy_params,
    column_to_tensor,
    convert_flux_to_mag,
    table_to_dict,
)
from bliss.reporting import get_single_galaxy_ellipticities


def _setup_single_galaxy_draw_generator(catalog_file: str, slen: int, seed: int):
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_file)

    stamp_size = slen * PIXEL_SCALE  # arcsecs

    sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=1,
        min_number=1,
        stamp_size=stamp_size,
        max_shift=0.0,
        min_mag=0,  # min mag in i-band is 14.32
        max_mag=27.3,  # see document of high level responses
        seed=seed,
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


class SingleGalsimGalaxies(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        catalog_file: str,  # should point to 'OneSqDeg.fits'
        num_workers: int,
        batch_size: int,
        n_batches: int,
        slen: int,
        seed: int,  # for draw generator
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()
        self.catalog_file = catalog_file
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = valid_n_batches
        self.seed = seed
        self.slen = slen

        sky_level: float = mean_sky_level("LSST", "i").to_value("electron")
        self.background = ConstantBackground((sky_level,))
        self.draw_generator = _setup_single_galaxy_draw_generator(
            self.catalog_file, self.slen, self.seed
        )

    def __getitem__(self, idx):
        batch = next(self.draw_generator)
        galaxy_image = batch.blend_images[0, None, 3]  # '3' refers to i-band
        background = self.background.sample((1, *galaxy_image.shape)).squeeze(1)
        return {
            "images": _add_noise_and_background(galaxy_image, background),
            "background": background,
            "noiseless": galaxy_image,
            "params": table_to_dict(batch.catalog_list[0]),
            "snr": _get_snr(galaxy_image, background),
        }

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        dl = DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)
        if not self.fix_validation_set:
            return dl
        valid: List[Dict[str, Tensor]] = []
        for _ in tqdm(range(self.valid_n_batches), desc="Generating fixed validation set"):
            valid.append(next(iter(dl)))
        return DataLoader(valid, batch_size=None, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)


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
        max_shift=slen * PIXEL_SCALE,
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


class GalsimBlends(pl.LightningDataModule, Dataset):
    """Dataset of galsim blends."""

    def __init__(
        self,
        catalog_file: str,  # should point to 'OneSqDeg.fits'
        stars_file: str,  # should point to 'stars_med_june2018.fits'
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        bp: int,
        slen: int,
        seed: int,  # for draw generator
        star_density: float = 10,  # counts / sq. arcmin
        galaxy_density: float = 185,  # counts / sq. arcmin
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = valid_n_batches
        self.seed = seed

        # images
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.bp = bp
        self.slen = slen
        self.size = self.slen + 2 * self.bp
        self.pixel_scale = PIXEL_SCALE

        # counts and densities
        self.galaxy_density = galaxy_density
        self.star_density = star_density

        # we just need something sensible here
        self.max_n_sources = math.ceil(galaxy_density * (self.slen * PIXEL_SCALE / 60) ** 2 * 2)
        self.max_n_stars = self.max_n_sources * (self.star_density / self.galaxy_density)

        sky_level: float = mean_sky_level("LSST", "i").to_value("electron")
        self.background = ConstantBackground((sky_level,))

        # btk
        self.galaxy_generator = _setup_blend_galaxy_generator(
            catalog_file, self.galaxy_density, self.max_n_sources, self.slen, self.bp, self.seed
        )

        self.all_star_magnitudes = column_to_tensor(Table.read(stars_file), "i_ab")

    def sample(self):
        # sampling
        star_params = sample_stars(
            self.all_star_magnitudes, self.slen, self.star_density, self.max_n_sources
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
        star_bools = (1 - galaxy_bools) * is_on_sources

        # star fluxes and log_fluxes
        star_fluxes = star_params["star_fluxes"]
        star_log_fluxes = star_params["star_log_fluxes"]
        new_star_fluxes = torch.zeros((self.max_n_sources, 2))
        new_star_log_fluxes = torch.zeros((self.max_n_sources, 2))
        new_star_fluxes[n_galaxies : n_galaxies + n_stars, :] = star_fluxes[:n_stars]
        new_star_log_fluxes[n_galaxies : n_galaxies + n_stars, :] = star_log_fluxes[:n_stars]

        # galaxy params
        galaxy_params = catsim_row_to_galaxy_params(blend_cat, self.max_n_sources)

        # images
        stars_image, stars_isolated = render_stars(
            n_stars, star_fluxes, star_locs, self.slen, self.bp, psf, self.max_n_stars
        )
        galaxies_image = batch.blend_images[0, 3, None]
        galaxies_isolated = batch.blend_images[0, :, 3, None]
        noiseless = stars_image + galaxies_image
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
            "star_fluxes": star_fluxes,
            "star_log_fluxes": star_log_fluxes,
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
        gal_fluxes = galaxy_params[0, :, 0]
        star_fluxes = full_cat["star_fluxes"][0, :, 0]
        mags = torch.zeros(self.max_n_sources)
        fluxes = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            if galaxy_bools[0, ii, 0].item() == 1:
                mags[ii] = convert_flux_to_mag(gal_fluxes[ii]).item()
                fluxes[ii] = gal_fluxes[ii].item()
            else:
                mags[ii] = convert_flux_to_mag(star_fluxes[ii]).item()
                fluxes[ii] = star_fluxes[ii].item()
        mags = rearrange(mags, "n -> 1 n 1", n=self.max_n_sources)
        fluxes = rearrange(fluxes, "n -> 1 n 1", n=self.max_n_sources)

        # add to full catalog (needs to be in batches)
        full_cat["mags"] = mags
        full_cat["ellips"] = ellips
        full_cat["snr"] = snr
        full_cat["blendedness"] = blendedness
        full_cat["fluxes"] = fluxes
        return full_cat

    def _get_tile_params(self, full_cat):
        # since uniformly place galaxies in image, no hard upper limit on n_sources per tile.
        tile_cat = full_cat.to_tile_params(
            self.tile_slen, self.max_sources_per_tile, ignore_extra_sources=True
        )
        tile_dict = tile_cat.to_dict()
        n_sources = tile_dict.pop("n_sources")
        n_sources = rearrange(n_sources, "1 nth ntw -> nth ntw")

        return {
            "n_sources": n_sources,
            **{k: rearrange(v, "1 nth ntw n d -> nth ntw n d") for k, v in tile_dict.items()},
        }

    def _run_nan_check(self, *tensors):
        for t in tensors:
            assert not torch.any(torch.isnan(t))

    def __getitem__(self, idx):
        full_cat, images, noiseless, isolated_images, bg, psf = self.sample()
        full_cat = self._add_metrics(full_cat, noiseless, isolated_images, bg, psf)
        tile_params = self._get_tile_params(full_cat)
        self._run_nan_check(images, bg, *tile_params.values())
        return {"images": images, "background": bg, **tile_params}

    def __len__(self):
        return self.batch_size * self.n_batches

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        dl = DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)
        if not self.fix_validation_set:
            return dl
        valid: List[Dict[str, Tensor]] = []
        for _ in tqdm(range(self.valid_n_batches), desc="Generating fixed validation set"):
            valid.append(next(iter(dl)))
        return DataLoader(valid, batch_size=None, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers)


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
