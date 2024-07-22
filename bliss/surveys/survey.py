from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class Survey(pl.LightningDataModule, Dataset, ABC):
    BANDS = ()

    def __init__(self):
        super().__init__()

        self.align_to_band = None

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

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    @abstractmethod
    def image_id(self, idx: int):
        """Return the image_id for the given index."""

    @abstractmethod
    def idx(self, image_id):
        """Return the index for the given image_id."""

    @abstractmethod
    def image_ids(self) -> list:
        """Return a list of all image_ids."""

    def predict_dataloader(self):
        """Return a DataLoader for prediction."""
        return DataLoader(SurveyPredictIterator(self), batch_size=1)


class SurveyPredictIterator:
    def __init__(self, survey):
        self.survey = survey

    def __getitem__(self, idx):
        x = self.survey[idx]
        return {"images": x["image"], "psf_params": x["psf_params"]}

    def __len__(self):
        return len(self.survey)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]


class SurveyDownloader:
    def download_catalog(self, **kwargs) -> str:
        """Download the catalog and return the path to the catalog file."""
        raise NotImplementedError
