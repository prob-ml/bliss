from typing import Optional, Dict, List

import galsim
import torch
from torch import Tensor
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm

from bliss.catalog import FullCatalog
from bliss.datasets.sdss import convert_flux_to_mag
from bliss.reporting import get_single_galaxy_ellipticities
from bliss.datasets.background import ConstantBackground
from case_studies.psf_homogenization.homogenization import psf_homo
from case_studies.psf_homogenization.galsim_star import UniformGalsimPrior, FullCatelogDecoderSG
from case_studies.psf_homogenization.psf_decoder import PsfSampler


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise

def _get_snr(image: Tensor, background: Tensor) -> float:
    image_with_background = image + background
    return torch.sqrt(torch.sum(image**2 / image_with_background)).item()

def _get_blendedness(single_galaxy: Tensor, all_galaxies: Tensor) -> float:
    num = torch.sum(single_galaxy * single_galaxy).item()
    denom = torch.sum(single_galaxy * all_galaxies).item()
    return 1 - num / denom

class GalsimBlendsSGRand(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        prior: UniformGalsimPrior,
        decoder: FullCatelogDecoderSG,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        psf_sampler: PsfSampler,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.background = background
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = valid_n_batches

        self.max_n_sources = self.prior.max_n_sources
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.bp = self.decoder.bp
        self.slen = self.decoder.slen
        self.pixel_scale = self.decoder.single_galaxy_decoder.pixel_scale
        self.psf = psf_sampler

    def _get_images(self, full_cat):
        psf_obj = self.psf.sample()
        noiseless, noiseless_centered, noiseless_uncentered = self.decoder.render_catalog(
            full_cat, psf_obj
        )

        # get background and noisy image
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        return (  # noqa: WPS227
            noisy_image,
            noiseless,
            noiseless_centered,
            noiseless_uncentered,
            background,
        )

    def __getitem__(self, idx):
        full_cat = self._sample_full_catalog()
        images, *metric_images, background = self._get_images(full_cat)
        full_cat = self._add_metrics(full_cat, *metric_images, background)
        tile_params = self._get_tile_params(full_cat)
        return {
            "images": images,
            "background": background,
            **tile_params,
        }
    
    def _sample_full_catalog(self):
        params_dict = self.prior.sample()
        params_dict["plocs"] = params_dict["locs"] * self.slen
        params_dict.pop("locs")
        params_dict = {k: v.unsqueeze(0) for k, v in params_dict.items()}
        return FullCatalog(self.slen, self.slen, params_dict)
    
    def _add_metrics(
        self,
        full_cat: FullCatalog,
        noiseless: Tensor,
        noiseless_centered: Tensor,
        noiseless_uncentered: Tensor,
        background: Tensor,
    ):
        n_sources = int(full_cat.n_sources.item())
        galaxy_params = full_cat["galaxy_params"][0]

        # add important metrics to full catalog
        scale = self.pixel_scale
        size = self.slen + 2 * self.bp
        psf = self.decoder.single_galaxy_decoder.psf  # psf from single galaxy decoder.
        psf_tensor = torch.from_numpy(psf.drawImage(nx=size, ny=size, scale=scale).array)

        single_galaxy_tensor = noiseless_centered[:n_sources]
        single_galaxy_tensor = rearrange(single_galaxy_tensor, "n 1 h w -> n h w", n=n_sources)
        mags = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            mags[ii] = convert_flux_to_mag(galaxy_params[ii, 0])
        ellips = torch.zeros(self.max_n_sources, 2)
        e12 = get_single_galaxy_ellipticities(single_galaxy_tensor, psf_tensor, scale)
        ellips[:n_sources, :] = e12

        # get snr and blendedness
        snr = torch.zeros(self.max_n_sources)
        blendedness = torch.zeros(self.max_n_sources)
        for ii in range(n_sources):
            snr[ii] = _get_snr(noiseless_centered[ii], background)
            blendedness[ii] = _get_blendedness(noiseless_uncentered[ii], noiseless)

        gal_fluxes = galaxy_params[:, 0]

        # add to full catalog (needs to be in batches)
        full_cat["mags"] = rearrange(mags, "n -> 1 n 1", n=self.max_n_sources)
        full_cat["fluxes"] = torch.zeros(1, self.max_n_sources, 1)  # stars
        full_cat["log_fluxes"] = torch.zeros(1, self.max_n_sources, 1)  # stars
        full_cat["galaxy_fluxes"] = rearrange(gal_fluxes, "n -> 1 n 1", n=self.max_n_sources)

        full_cat["ellips"] = rearrange(ellips, "n g -> 1 n g", n=self.max_n_sources, g=2)
        full_cat["snr"] = rearrange(snr, "n -> 1 n 1", n=self.max_n_sources)
        full_cat["blendedness"] = rearrange(blendedness, "n -> 1 n 1", n=self.max_n_sources)

        return full_cat
    
    def _get_tile_params(self, full_cat):
        tile_cat = full_cat.to_tile_params(
            self.tile_slen, self.max_sources_per_tile, ignore_extra_sources=True
        )  # since uniformly place galaxies in image, no hard upper limit on n_sources per tile.
        tile_dict = tile_cat.to_dict()
        n_sources = tile_dict.pop("n_sources")
        n_sources = rearrange(n_sources, "1 nth ntw -> nth ntw")

        return {
            "n_sources": n_sources,
            **{k: rearrange(v, "1 nth ntw n d -> nth ntw n d") for k, v in tile_dict.items()},
        }
    
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

    def __len__(self):
        return self.batch_size * self.n_batches


class GalsimBlendsSGwithPSF(GalsimBlendsSGRand):
    def __init__(
        self,
        prior: UniformGalsimPrior,
        decoder: FullCatelogDecoderSG,
        background: ConstantBackground,
        tile_slen: int,
        max_sources_per_tile: int,
        num_workers: int,
        batch_size: int,
        n_batches: int,
        psf_sampler: PsfSampler,
        std_psf_fwhm: float = 1.0,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__(
            prior,
            decoder,
            background,
            tile_slen,
            max_sources_per_tile,
            num_workers,
            batch_size,
            n_batches,
            psf_sampler,
            fix_validation_set,
            valid_n_batches
        )
        self.std_psf = torch.from_numpy(
            galsim.Gaussian(fwhm=std_psf_fwhm)
            .drawImage(nx=self.slen, ny=self.slen, scale=self.pixel_scale)  # noqa: WPS348
            .array  # noqa: WPS348
        )

    def _get_images(self, full_cat):
        psf_obj = self.psf.sample()
        noiseless, noiseless_centered, noiseless_uncentered = self.decoder.render_catalog(
            full_cat, psf_obj
        )

        # get background and noisy image
        background = self.background.sample((1, *noiseless.shape)).squeeze(0)
        noisy_image = _add_noise_and_background(noiseless, background)

        # homogenization
        std_psf = self.std_psf.reshape(1, 1, self.slen, self.slen)
        psf = torch.from_numpy(
            psf_obj.drawImage(nx=self.slen, ny=self.slen, scale=self.pixel_scale).array
        ).reshape(1, 1, self.slen, self.slen)
        homo_image, _ = psf_homo(
            noisy_image.reshape(1, *noisy_image.shape),
            psf,
            std_psf,
            background.reshape(1, *noisy_image.shape),
        )

        return (  # noqa: WPS227
            noisy_image,
            noiseless,
            noiseless_centered,
            noiseless_uncentered,
            background,
            homo_image.reshape(*noisy_image.shape),
            psf,
        )

    def __getitem__(self, idx):
        full_cat = self._sample_full_catalog()
        images, *metric_images, background, homo_image, psf = self._get_images(full_cat)
        full_cat = self._add_metrics(full_cat, *metric_images, background)
        tile_params = self._get_tile_params(full_cat)
        return {
            "images": homo_image,
            "background": background,
            "noisy_image": images,
            "psf": psf,
            "std_psf": self.std_psf,
            **tile_params,
        }
