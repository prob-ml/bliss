from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class Survey(pl.LightningDataModule, Dataset, ABC):
    BANDS = ()

    def __init__(self):
        super().__init__()

        self.align_to_band = None
        self.crop_hw = None
        self.crop_bands = None

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
        return DataLoader(SurveyPredictIterator(self, self.crop_bands, self.crop_hw), batch_size=1)


class SurveyPredictIterator:
    def __init__(self, survey, crop_bands=None, crop_hw=None):
        self.survey = survey
        self.crop_bands = crop_bands  # includes all bands if None
        self.crop_hw = crop_hw

    @classmethod
    def crop_to_mult16(cls, x):
        """Crop the image dimensions to a multiple of 16."""
        # note: by cropping the top-right, we preserve the mapping between pixel coordinates
        # and the original WCS coordinates
        height = x.shape[1] - (x.shape[1] % 16)
        width = x.shape[2] - (x.shape[2] % 16)
        return x[:, :height, :width]

    def __getitem__(self, idx):
        item = self.survey[idx]

        # back to physical units (and assuming image is sky subtracted)
        images = item["image"] / item["flux_calibration"]

        if self.crop_bands is not None:
            images = images[self.crop_bands]

        if self.crop_hw is not None:
            r1, r2, c1, c2 = self.crop_hw
            images = images[:, r1:r2, c1:c2]

        images = self.crop_to_mult16(images)  # alternatively, could pad

        return {"images": images, "psf_params": item["psf_params"]}

    def __len__(self):
        return len(self.survey)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]


class SurveyDownloader:
    def download_catalog(self, **kwargs) -> str:
        """Download the catalog and return the path to the catalog file."""
        raise NotImplementedError
