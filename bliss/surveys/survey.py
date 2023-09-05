from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset


class Survey(pl.LightningDataModule, Dataset, ABC):
    BANDS = ()

    def __init__(self):
        super().__init__()

        self.bands = None
        self.background = None
        self.psf = None
        self.flux_calibration_dict = None
        self.pixel_shift = None

        self.catalog_cls = None  # TODO: better way than `survey.catalog_cls`?

    @staticmethod
    def coadd_images(constituent_images):
        """Coadd the constituent images into a single image."""
        raise NotImplementedError

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
    def image_ids(self) -> list:
        """Return a list of all image_ids."""

    @property
    @abstractmethod
    def predict_batch(self):
        """Return a batch of data for prediction."""

    @predict_batch.setter
    @abstractmethod
    def predict_batch(self):
        """Set a batch of data for prediction."""

    def get_flux_calibrations(self):
        d = {}
        for i, image_id in enumerate(self.image_ids()):
            nelec_conv_for_frame = self[i]["flux_calibration_list"]
            avg_nelec_conv = np.squeeze(np.mean(nelec_conv_for_frame, axis=1))
            d[image_id] = avg_nelec_conv
        return d


class SurveyDownloader:
    def download_catalog(self, **kwargs) -> str:
        """Download the catalog and return the path to the catalog file."""
        raise NotImplementedError
