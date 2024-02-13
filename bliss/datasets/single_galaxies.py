from typing import Dict, List, Optional

import btk
import pytorch_lightning as pl
import torch
from galcheat.utilities import mean_sky_level
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bliss.datasets.background import ConstantBackground
from bliss.datasets.lsst import PIXEL_SCALE, table_to_dict


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
        galaxy_image_np = batch.blend_images[0, None, 3]  # '3' refers to i-band
        galaxy_image = torch.from_numpy(galaxy_image_np)
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


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _get_snr(image: Tensor, background: Tensor) -> float:
    image_with_background = image + background
    return torch.sqrt(torch.sum(image**2 / image_with_background)).item()
