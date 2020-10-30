from omegaconf import DictConfig
import torch
from torch.utils.data import IterableDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bliss.models.decoder import ImageDecoder


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


class SimulatedModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(SimulatedModule, self).__init__()
        self.cfg = cfg
        self.dataset = SimulatedDataset(cfg)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)
