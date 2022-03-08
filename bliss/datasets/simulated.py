import warnings
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

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
        background: Tuple[float, ...],
        n_batches=10,
        batch_size=32,
        generate_device="cpu",
        testing_file=None,
    ):
        super().__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.image_prior = prior.to(generate_device)
        self.image_prior.requires_grad_(False)  # freeze decoder weights.
        self.image_decoder = decoder.to(generate_device)
        self.image_decoder.requires_grad_(False)  # freeze decoder weights.
        self.testing_file = testing_file
        self.background = torch.tensor(background, device=generate_device)

        # check sleep training will work.
        n_tiles_per_image = self.image_prior.n_tiles_h * self.image_prior.n_tiles_w
        total_ptiles = n_tiles_per_image * self.batch_size
        assert total_ptiles > 1, "Need at least 2 tiles over all batches."

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        with torch.no_grad():
            batch = self.image_prior.sample_prior(batch_size=self.batch_size)
            images = self.image_decoder.render_images(
                batch["n_sources"],
                batch["locs"],
                batch["galaxy_bools"],
                batch["galaxy_params"],
                batch["fluxes"],
            )
            background = self.make_background(*images.shape)
            images += background
            images = self._apply_noise(images)
            batch.update(
                {
                    "images": images,
                    "background": background,
                }
            )

        return batch

    def make_background(self, batch_size, c, hlen, wlen):
        return self.background.reshape(1, c, 1, 1).expand(batch_size, -1, hlen, wlen)

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean

        return images

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def test_dataloader(self):
        dl = DataLoader(self, batch_size=None, num_workers=0)

        if self.testing_file:
            test_dataset = BlissDataset(self.testing_file)
            dl = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

        return dl


class BlissDataset(Dataset):
    """A dataset created from simulated batches saved as a single dict by bin/generate.py."""

    def __init__(self, pt_file="example.pt"):
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
