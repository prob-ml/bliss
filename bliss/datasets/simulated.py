from omegaconf import DictConfig
import torch
from torch.utils.data import IterableDataset, Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bliss.models.decoder import ImageDecoder


class SimulatedModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(SimulatedModule, self).__init__()
        self.cfg = cfg
        self.dataset = SimulatedDataset(cfg)

        # check datamodule is sensible for sleep training.
        n_tiles_per_image = self.dataset.image_decoder.n_tiles_per_image
        total_ptiles = n_tiles_per_image * self.dataset.batch_size
        assert total_ptiles > 1, "Need at least 2 tiles in each batch."

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def test_dataloader(self):
        if self.cfg.testing.file is None:
            return DataLoader(self.dataset, batch_size=None)

        else:
            test_dataset = SavedSimulated(self.cfg.testing.file)
            batch_size = self.cfg.testing.batch_size
            return DataLoader(test_dataset, batch_size=batch_size, num_workers=0)


class SimulatedDataset(IterableDataset):
    def __init__(self, cfg: DictConfig):
        super(SimulatedDataset, self).__init__()

        self.n_batches = cfg.dataset.params.n_batches
        self.batch_size = cfg.dataset.params.batch_size
        self.image_decoder = ImageDecoder(**cfg.model.decoder.params)
        self.image_decoder.requires_grad_(False)

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for i in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        with torch.no_grad():
            batch = self.image_decoder.sample_prior(batch_size=self.batch_size)
            images = self.image_decoder.render_images(
                batch["n_sources"],
                batch["locs"],
                batch["galaxy_bool"],
                batch["galaxy_params"],
                batch["fluxes"],
            )
            batch.update(
                {"images": images, "background": self.image_decoder.background}
            )

        return batch


class SavedSimulated(Dataset):
    def __init__(self, pt_file="example.pt"):
        """A dataset created from simulated batches saved as a single dict created from
        SimulatedDataset. Hint: Create a single big batch."""
        super().__init__()

        self.data = torch.load(pt_file)
        assert isinstance(self.data, dict)
        self.background = self.data.pop("background")
        self.size = self.data["images"].shape[0]

    def __len__(self):
        """Number of batches saved in the file."""
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
