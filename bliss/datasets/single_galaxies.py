from typing import Dict

import btk
import torch
from galcheat.utilities import mean_sky_level
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

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


class SingleGalsimGalaxies(Dataset):
    def __init__(
        self,
        catalog_file: str,  # should point to 'OneSqDeg.fits'
        slen: int,
        seed: int,  # for draw generator
    ):
        super().__init__()
        self.catalog_file = catalog_file
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

        params = table_to_dict(batch.catalog_list[0])
        params = {k: v.unsqueeze(0) for k, v in params.items()}

        d = {
            "images": _add_noise_and_background(galaxy_image, background).unsqueeze(0),
            "background": background.unsqueeze(0),
            "noiseless": galaxy_image.unsqueeze(0),
            "params": params,
            "snr": torch.tensor(_get_snr(galaxy_image, background)).unsqueeze(0),
        }

        return TensorDict(d, batch_size=[1])


class SavedSingleGalsimGalaxies(Dataset):
    def __init__(self, dataset_file: str, epoch_size: int) -> None:
        super().__init__()
        ds = _load_dataset(dataset_file)
        assert {"images", "background", "noiseless", "params", "snr"}.issubset(ds)

        self.ds = ds
        self.epoch_size = epoch_size

        assert len(self.ds["images"]) == self.epoch_size, "Train on entired saved dataset."

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index) -> Dict[str, Tensor]:
        return {k: v[index] for k, v in self.ds.items()}


def _load_dataset(file: str):
    return torch.load(file)


def _add_noise_and_background(image: Tensor, background: Tensor) -> Tensor:
    image_with_background = image + background
    noise = image_with_background.sqrt() * torch.randn_like(image_with_background)
    return image_with_background + noise


def _get_snr(image: Tensor, background: Tensor) -> float:
    image_with_background = image + background
    return torch.sqrt(torch.sum(image**2 / image_with_background)).item()
