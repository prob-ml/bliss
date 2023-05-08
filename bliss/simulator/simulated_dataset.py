import os
import pickle
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.generate import FileDatum
from bliss.simulator.background import ConstantBackground, SimulatedSDSSBackground
from bliss.simulator.decoder import ImageDecoder
from bliss.simulator.prior import ImagePrior

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        prior: ImagePrior,
        decoder: ImageDecoder,
        background: Union[ConstantBackground, SimulatedSDSSBackground],
        n_batches: int,
        num_workers: int = 0,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()

        self.n_batches = n_batches

        self.image_prior = prior
        self.image_decoder = decoder
        self.background = background
        self.image_prior.requires_grad_(False)
        self.background.requires_grad_(False)
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.
        assert torch.all(images_mean > 1e-8)
        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean
        return images

    def simulate_image(self, tile_catalog: TileCatalog) -> Tuple[Tensor, Tensor]:
        images = self.image_decoder.render_images(tile_catalog)
        background = self.background.sample(images.shape)
        images += background
        images = self._apply_noise(images)
        return images, background

    def get_batch(self) -> Dict:
        with torch.no_grad():
            tile_catalog = self.image_prior.sample_prior()
            images, background = self.simulate_image(tile_catalog)
            return {
                "tile_catalog": tile_catalog.to_dict(),
                "images": images,
                "background": background,
            }

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.fix_validation_set:
            valid: List[Dict[str, Tensor]] = []
            for _ in tqdm(range(self.valid_n_batches), desc="Generating fixed validation set"):
                valid.append(self.get_batch())
            num_workers = 0
        else:
            valid = self
            num_workers = self.num_workers
        return DataLoader(valid, batch_size=None, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)


class CachedSimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        n_batches: int,
        valid_n_batches: int,
        fix_validation_set: bool,
        num_workers: int,
        batch_size: int,
        file_data_capacity: int,
        cached_data_path: str,
    ):
        super().__init__()

        self.n_batches = n_batches
        self.valid_n_batches = valid_n_batches
        self.fix_validation_set = fix_validation_set
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.file_data_capacity = file_data_capacity
        self.cached_data_path = cached_data_path

        self.data: List[FileDatum] = []

        # assume cached image files exist, read from disk
        for filename in os.listdir(self.cached_data_path):
            if "valid" in filename:
                continue
            if filename.startswith("dataset") and filename.endswith(".pkl"):
                self.data += self.read_file(f"{self.cached_data_path}/{filename}")
        assert self.data, "No cached data loaded; run `generate.py` first"

    def read_file(self, filename: str) -> List[FileDatum]:
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_batch(self, batch_idx) -> Dict:
        batch_data = self.data[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        images = torch.stack([datum["images"] for datum in batch_data])
        background = torch.stack([datum["background"] for datum in batch_data])
        tile_catalog = {}
        assert len(batch_data), f"Batch for batch_idx {batch_idx} is empty"
        for key in batch_data[0]["tile_catalog"]:
            tile_catalog[key] = torch.stack([datum["tile_catalog"][key] for datum in batch_data])
        return {
            "tile_catalog": tile_catalog,
            "images": images,
            "background": background,
        }

    def __iter__(self):
        random.shuffle(self.data)
        for batch_idx in range(self.n_batches):
            yield self.get_batch(batch_idx)

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.fix_validation_set:
            assert os.path.exists(
                f"{self.cached_data_path}/dataset_valid.pkl"
            ), "No cached validation data found; run `generate.py` with 'fix_validation_set=true'"
            valid = self.read_file(f"{self.cached_data_path}/dataset_valid.pkl")
        else:
            valid = self
        return DataLoader(valid, batch_size=None, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)
