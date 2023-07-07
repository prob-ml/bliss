from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import Dataset


class Survey(pl.LightningDataModule, Dataset, ABC):
    def __init__(
        self,
        predict_device=None,
        predict_crop=None,
    ):
        super().__init__()

        self.predict_device = predict_device
        self.predict_crop = predict_crop

        self.bands = None
        self.prior = None
        self.background = None
        self.psf = None

    @abstractmethod
    def prepare_data(self):
        """pl.LightningDataModule override."""

    @abstractmethod
    def __len__(self):
        """Dataset override."""

    @abstractmethod
    def __getitem__(self, idx):
        """Dataset override."""

    @abstractmethod
    def image_id(self, idx: int):
        """Return the image_id for the given index."""

    @abstractmethod
    def idx(self, image_id):
        """Return the index for the given image_id."""

    @abstractmethod
    def image_ids(self):
        """Return a list of all image_ids."""

    @abstractmethod
    def predict_dataloader(self):
        """pl.LightningDataModule override."""
