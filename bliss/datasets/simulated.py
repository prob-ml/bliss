from omegaconf import DictConfig
import pytorch_lightning as pl
import warnings

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader

from bliss.models.decoder import ImageDecoder

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(self, cfg: DictConfig):
        super(SimulatedDataset, self).__init__()
        self.cfg = cfg

        self.n_batches = cfg.dataset.params.n_batches
        self.batch_size = cfg.dataset.params.batch_size
        self.image_decoder = ImageDecoder(**cfg.model.decoder.params)
        self.image_decoder.requires_grad_(False)  # freeze decoder weights.

        # check sleep training will work.
        n_tiles_per_image = self.image_decoder.n_tiles_per_image
        total_ptiles = n_tiles_per_image * self.batch_size
        assert total_ptiles > 1, "Need at least 2 tiles over all batches."

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for i in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        with torch.no_grad():
            batch = self.image_decoder.sample_prior(batch_size=self.batch_size)
            images, _ = self.image_decoder.render_images(
                batch["n_sources"],
                batch["locs"],
                batch["galaxy_bool"],
                batch["galaxy_params"],
                batch["fluxes"],
                add_noise=True,
            )
            batch.update(
                {
                    "images": images,
                    "background": self.image_decoder.background,
                    "slen": torch.tensor([self.image_decoder.slen]),
                }
            )

        return batch

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def test_dataloader(self):
        dl = DataLoader(self, batch_size=None, num_workers=0)

        if self.cfg.testing.file is not None:
            test_dataset = BlissDataset(self.cfg.testing.file)
            batch_size = self.cfg.testing.batch_size
            num_workers = self.cfg.testing.num_workers
            dl = DataLoader(
                test_dataset, batch_size=batch_size, num_workers=num_workers
            )

        return dl


class BlissDataset(Dataset):
    def __init__(self, pt_file="example.pt"):
        """A dataset created from simulated batches saved as a single dict by
        bin/generate.py"""
        super(BlissDataset, self).__init__()

        data = torch.load(pt_file)
        assert isinstance(data, dict)

        self.data = data
        self.size = self.data["images"].shape[0]
        self.background = self.data.pop("background")
        self.slen = self.data.pop("slen")

    def __len__(self):
        """Number of batches saved in the file."""
        return self.size

    def __getitem__(self, idx):
        d = {k: v[idx] for k, v in self.data.items()}
        d.update({"background": self.background, "slen": self.slen})
        return d
