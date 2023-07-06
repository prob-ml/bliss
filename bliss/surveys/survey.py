import pytorch_lightning as pl
from torch.utils.data import Dataset


class Survey(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        predict_device=None,
        predict_crop=None,
    ):
        super().__init__()

        self.predict_device = predict_device
        self.predict_crop = predict_crop

        self._prior = None
        self._background = None
        self._psf = None

    def prepare_data(self):
        """pl.LightningDataModule override."""
        return NotImplemented

    def __len__(self):
        """Dataset override."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Dataset override."""
        return NotImplemented

    def image_id(self, idx: int):
        """Return the image_id for the given index."""
        return NotImplemented

    def idx(self, image_id):
        """Return the index for the given image_id."""
        return NotImplemented

    def image_ids(self):
        """Return a list of all image_ids."""
        return NotImplemented

    def predict_dataloader(self):
        """pl.LightningDataModule override."""
        return NotImplemented

    @property
    def prior(self):
        """Get the prior."""
        return self._prior

    @prior.setter
    def prior(self, prior):
        """Set the prior."""
        self._prior = prior

    @property
    def background(self):
        """Get the image background."""
        return self._background

    @background.setter
    def background(self, background):
        """Set the image background."""
        self._background = background

    @property
    def psf(self):
        """Get the image PSF."""
        return self._psf

    @psf.setter
    def psf(self, psf):
        """Set the image PSF."""
        self._psf = psf
