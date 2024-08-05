from abc import ABC, abstractmethod

import pytorch_lightning as pl
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from bliss.align import align


class Survey(pl.LightningDataModule, Dataset, ABC):
    BANDS = ()

    def __init__(self):
        super().__init__()

        self.align_to_band = None
        self.crop_to_hw = None
        self.crop_to_bands = None

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
        survey_iterator = SurveyPredictIterator(self)
        return DataLoader(survey_iterator, batch_size=1)


class SurveyPredictIterator:
    def __init__(self, survey):
        self.survey = survey

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
        images = item["image"]

        # assume the images are already sky subtracted if no background is provided
        images -= item.get("background", 0.0)

        # back to physical units (and assuming image is sky subtracted)
        images /= rearrange(item["flux_calibration"], "bands w -> bands 1 w")

        # alignment is done after cropping here for speed, mainly during testing,
        # but this may not be a ideal in general
        if self.survey.align_to_band is not None:
            images = align(images, wcs_list=item["wcs"], ref_band=self.survey.align_to_band)

        # includes all bands if None
        if self.survey.crop_to_bands is not None:
            images = images[self.survey.crop_to_bands]
            item["psf_params"] = item["psf_params"][self.survey.crop_to_bands]

        if self.survey.crop_to_hw is not None:
            r1, r2, c1, c2 = self.survey.crop_to_hw
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
