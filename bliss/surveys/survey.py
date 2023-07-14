from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import Dataset


class Survey(pl.LightningDataModule, Dataset, ABC):
    def __init__(self):
        super().__init__()

        self.bands = None
        self.prior = None
        self.background = None
        self.psf = None

        self.catalog_cls = None  # TODO: better way than `survey.catalog_cls`?

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

    @property
    @abstractmethod
    def predict_batch(self):
        """Return a batch of data for prediction."""

    @predict_batch.setter
    @abstractmethod
    def predict_batch(self):
        """Set a batch of data for prediction."""


class SurveyDownloader:
    def download_catalog(self, **kwargs) -> str:
        """Download the catalog and return the path to the catalog file."""
        raise NotImplementedError
