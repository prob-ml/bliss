from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bliss.catalog import FullCatalog
from bliss.simulator.background import ConstantBackground
from bliss.simulator.galsim_decoder import (
    DefaultGalsimPrior,
    FullCatalogDecoder,
    SingleGalsimGalaxyPrior,
)
from bliss.simulator.galsim_galaxies import GalsimBlends
from case_studies.coadds.align import align_single_exposures


def load_coadd_dataset(path: str) -> Tuple[dict, FullCatalog]:
    test_ds = torch.load(path)
    image_keys = {"coadd_5", "coadd_10", "coadd_25", "coadd_35", "coadd_50", "single"}
    all_keys = list(test_ds.keys())
    truth_params = {k: test_ds.pop(k) for k in all_keys if k not in image_keys}
    truth_params["n_sources"] = truth_params["n_sources"].reshape(-1)
    truth_cat = FullCatalog(40, 40, truth_params)
    return test_ds, truth_cat


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _linear_coadd(aligned_images, weight):
    num = torch.sum(torch.mul(weight, aligned_images), dim=0)
    return num / torch.sum(weight, dim=0)


class CoaddDefaultGalsimPrior(DefaultGalsimPrior):
    def __init__(
        self,
        single_galaxy_prior: SingleGalsimGalaxyPrior,
        max_n_sources: int,
        mean_sources: float,
        max_shift: float,
        galaxy_prob: float,
        n_dithers: int,
    ):
        super().__init__(
            single_galaxy_prior,
            max_n_sources,
            mean_sources,
            max_shift,
            galaxy_prob,
        )
        self.n_dithers = n_dithers

    def sample(self) -> Dict[str, Tensor]:
        """Returns a single batch of source parameters."""
        d = super().sample()
        d["dithers"] = torch.distributions.uniform.Uniform(-0.5, 0.5).sample([self.n_dithers, 2])
        return d


class CoaddFullCatalogDecoder(FullCatalogDecoder):
    def render_catalog(self, full_cat: FullCatalog, dithers: Tensor) -> Tuple[Tensor, Tensor]:
        size = self.slen + 2 * self.bp
        images = torch.zeros(len(dithers), 1, size, size)
        plocs0 = full_cat.plocs.clone()
        image0, _, _ = super().render_catalog(full_cat)
        for ii, dth in enumerate(dithers):
            full_cat.plocs = plocs0 + dth.reshape(1, 1, 2)
            image, _, _ = super().render_catalog(full_cat)
            images[ii] = image
        full_cat.plocs = plocs0
        return images, image0


class CoaddGalsimBlends(GalsimBlends):
    """Dataset of coadd galsim blends."""

    def sample_full_catalog(self):
        params_dict = self.prior.sample()
        dithers = params_dict["dithers"]
        params_dict.pop("dithers")
        params_dict["plocs"] = params_dict["locs"] * self.slen
        params_dict.pop("locs")
        params_dict = {k: v.unsqueeze(0) for k, v in params_dict.items()}
        return FullCatalog(self.slen, self.slen, params_dict), dithers

    def get_images(self, full_cat, dithers):
        size = self.slen + 2 * self.bp
        noiseless, image0 = self.decoder.render_catalog(full_cat, dithers)
        image0 = image0.reshape(size, size)
        aligned_images = align_single_exposures(image0, noiseless, size, dithers)
        background = self.background.sample(rearrange(aligned_images, "d h w -> d 1 h w").shape)
        aligned_images = rearrange(aligned_images, "d h w -> d 1 h w")
        weight = 1 / (aligned_images + background.clone().detach())
        noisy_aligned_image = _add_noise_and_background(aligned_images, background)
        coadded_image = _linear_coadd(noisy_aligned_image, weight)

        image0 = image0.reshape(1, 1, size, size)
        full_background = self.background.sample(image0.shape)
        single_exposure = _add_noise_and_background(image0, full_background)
        return noiseless, coadded_image, single_exposure[0, :, 1:-1, 1:-1], background[0]

    def __getitem__(self, idx):
        full_cat, dithers = self.sample_full_catalog()
        _, coadded_image, single_exposure, background = self.get_images(full_cat, dithers)
        tile_params = self._get_tile_params(full_cat)
        return {
            "images": coadded_image,
            "noisy": single_exposure,
            "background": background,
            **tile_params,
        }


class SavedCoadds(Dataset):
    def __init__(
        self,
        dataset_file: str,
        coadd_name: str,
        background: ConstantBackground,
        epoch_size: int,
    ):
        super().__init__()

        all_images, truth_cat = load_coadd_dataset(dataset_file)
        self.images = all_images[coadd_name]
        self.cat = truth_cat
        _, _, full_slen, _ = self.images.shape
        self.background = background.sample((1, 1, full_slen, full_slen))
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def _get_tile_params_at_idx(self, idx: int, cat: FullCatalog):
        # first convert to dict
        d = {**cat}
        d["plocs"] = cat.plocs
        max_sources = cat.max_sources

        # index into dict
        d = {k: v[idx].reshape(1, max_sources, -1) for k, v in d.items()}
        d["n_sources"] = torch.tensor([cat.n_sources[idx].item()])

        # reconvert to full catalog
        one_cat = FullCatalog(cat.height, cat.width, d)

        # now we can convert to tiles
        tile_one_cat = one_cat.to_tile_params(4, 1, ignore_extra_sources=True)

        return {k: v[0] for k, v in tile_one_cat.to_dict().items()}

    def __getitem__(self, idx):
        d = self._get_tile_params_at_idx(idx, self.cat)
        d.update({"background": self.background[0], "images": self.images[idx]})
        return d


class SavedCoaddsModule(pl.LightningDataModule):
    def __init__(self, train_ds: SavedCoadds, val_ds: SavedCoadds, batch_size: int):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
