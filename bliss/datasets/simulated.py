import warnings
from typing import Dict, List, Optional, Tuple, Union
from queue import Queue

import pytorch_lightning as pl
import os
import threading
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.datasets.background import ConstantBackground, SimulatedSDSSBackground
from bliss.models.decoder import ImageDecoder
from bliss.models.prior import ImagePrior

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
        n_tiles_h: int,
        n_tiles_w: int,
        n_batches: int,
        batch_size: int,
        generate_device: str,
        testing_file: Optional[str] = None,
        num_workers: int = 0,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.testing_file = testing_file
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w

        self.image_prior = prior
        self.image_decoder = decoder
        self.background = background
        self.image_prior.requires_grad_(False)
        self.image_decoder.requires_grad_(False)
        self.background.requires_grad_(False)

        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

        # check training will work.
        total_ptiles = self.batch_size * self.n_tiles_h * self.n_tiles_w
        assert total_ptiles > 1, "Need at least 2 tiles over all batches."

        torch.multiprocessing.set_start_method('spawn')
        self.setup_gpu = False
        BUF_SIZE = 2
        self.data_queue = Queue(BUF_SIZE)
        producer_thread = threading.Thread(target=self.populate_queue, args=())
        producer_thread.start()

    def populate_queue(self):
        if not self.setup_gpu:
            rank = os.getenv("LOCAL_RANK") 
            if rank is None:
                rank = "0"
            rank = int(rank)

            self.to(f"cuda:{rank}")
            torch.random.manual_seed(rank) # to have non-repeated datasets for gradients
            self.setup_gpu = True
        
        while True:
            with torch.no_grad():
                tile_catalog = self.sample_prior(self.batch_size, self.n_tiles_h, self.n_tiles_w)
                images, background = self.simulate_image_from_catalog(tile_catalog)
                self.data_queue.put({**tile_catalog.to_dict(), "images": images, "background": background})

    image_prior: ImagePrior
    image_decoder: ImageDecoder

    def to(self, generate_device):
        self.image_prior: ImagePrior = self.image_prior.to(generate_device)
        self.image_decoder: ImageDecoder = self.image_decoder.to(generate_device)
        self.background: Union[ConstantBackground, SimulatedSDSSBackground] = self.background.to(
            generate_device
        )

    def sample_prior(self, batch_size: int, n_tiles_h: int, n_tiles_w: int) -> TileCatalog:
        return self.image_prior.sample_prior(self.tile_slen, batch_size, n_tiles_h, n_tiles_w)

    def simulate_image_from_catalog(self, tile_catalog: TileCatalog) -> Tuple[Tensor, Tensor]:
        images = self.image_decoder.render_images(tile_catalog)
        background = self.background.sample(images.shape)
        images += background
        images = self._apply_noise(images)
        return images, background

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean

        return images

    @property
    def tile_slen(self) -> int:
        return self.image_decoder.tile_slen

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self) -> Dict[str, Tensor]:
        return self.data_queue.get()

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
        dl = DataLoader(self, batch_size=None, num_workers=self.num_workers)

        if self.testing_file is not None:
            test_dataset = BlissDataset(self.testing_file)
            dl = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

        return dl


class BlissDataset(Dataset):
    """A dataset created from simulated batches saved as a single dict by bin/generate.py."""

    def __init__(self, pt_file: str = "example.pt"):
        super().__init__()

        data = torch.load(pt_file)
        assert isinstance(data, dict)

        self.data = data
        self.size = self.data["images"].shape[0]
        self.background = self.data.pop("background")
        self.slen = self.data.pop("slen")

    def __len__(self):
        """Get the number of batches saved in the file."""
        return self.size

    def __getitem__(self, idx):
        d = {k: v[idx] for k, v in self.data.items()}
        d.update({"background": self.background, "slen": self.slen})
        return d
