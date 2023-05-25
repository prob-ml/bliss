import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf.listconfig import ListConfig
from skimage.restoration import richardson_lucy
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.generate import FileDatum
from bliss.simulator.background import ConstantBackground, SimulatedSDSSBackground
from bliss.simulator.decoder import ImageDecoder
from bliss.simulator.prior import ImagePrior

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        prior: ImagePrior,
        decoder: ImageDecoder,
        background: Union[ConstantBackground, SimulatedSDSSBackground],
        n_batches: int,
        sdss_fields: ListConfig,
        num_workers: int = 0,
        fix_validation_set: bool = False,
        valid_n_batches: Optional[int] = None,
    ):
        super().__init__()

        self.n_batches = n_batches
        self.image_prior = prior
        self.batch_size = self.image_prior.batch_size
        self.image_decoder = decoder
        self.background = background
        self.image_prior.requires_grad_(False)
        self.background.requires_grad_(False)
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

        # list of (run, camcol, field) tuples from config
        self.rcf_list = self._get_rcf_list(sdss_fields)

    def _get_rcf_list(self, sdss_fields):
        """Converts dict of row, camcol, field params into list of tuples."""
        rcf_list = []  # list of (run, camcol, field) pairs
        for rcf_params in sdss_fields["field_list"]:
            run = rcf_params["run"]
            camcol = rcf_params["camcol"]
            rcf_list.extend([(run, camcol, field) for field in rcf_params["fields"]])
        return np.array(rcf_list)

    def get_random_rcf(self, num_samples=1):
        """Get random run, camcol, field combination from loaded params.

        Args:
            num_samples (int, optional): number of random samples to get. Defaults to 1.

        Returns:
            Array of (row, camcol, field) pairs and index of each pair in self.rcf_list.
        """
        n = np.random.randint(len(self.rcf_list), size=(num_samples,), dtype=int)
        return self.rcf_list[n], n

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.
        assert torch.all(images_mean > 1e-8)
        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean
        return images

    def simulate_image(
        self, tile_catalog: TileCatalog, rcf_indices
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Simulate a batch of images.

        Args:
            tile_catalog (TileCatalog): The TileCatalog to render.
            rcf_indices: Indices of row/camcol/field in self.rcf_list to sample from.

        Returns:
            Tuple[Tensor, Tensor]: tuple of images and backgrounds
        """
        rcf = self.rcf_list[rcf_indices]
        images, psfs = self.image_decoder.render_images(tile_catalog, rcf)
        background = self.background.sample(images.shape, rcf_indices=rcf_indices)  # type: ignore
        images += background
        images = self._apply_noise(images)
        deconv_images = self.get_deconvolved_images(images, background, psfs)
        return images, background, deconv_images

    def get_deconvolved_images(self, images, backgrounds, psfs) -> Tensor:
        """Deconvolve the synthetic images with the psf used to generate them.

        Args:
            images (ndarray): batch of images
            backgrounds (ndarray): batch of backgrounds
            psfs (ndarray): batch of psfs

        Returns:
            Tensor: batch of deconvolved images
        """
        deconv_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            for band in range(self.image_prior.n_bands):
                deconv_images[i][band] = self.deconvolve_image(
                    images[i][band], backgrounds[i][band], psfs[i][band]
                )
        return torch.from_numpy(deconv_images)

    def deconvolve_image(self, image, background, psf, pad=5):
        """Deconvolve a single image.

        Args:
            image (Tensor): the image to deconvolve
            background (Tensor): background of the image (used for padding)
            psf (ndarray): the psf used to generate the image
            pad (int): the pad width (in pixels). Defaults to 5.

        Returns:
            ndarray: the deconvolved image, same size as the original
        """
        padded_image = np.pad(image, pad, mode="constant", constant_values=background.mean().item())
        normalized = padded_image / np.max(padded_image)
        deconv = richardson_lucy(normalized, psf.original.image.array)
        return deconv[pad:-pad, pad:-pad]

    def get_batch(self) -> Dict:
        """Get a batch of simulated images.

        The images are simulated by first generating a tile catalog from the prior, followed
        by choosing a random background and PSF and using those to generate the image. By default,
        the same row, camcol, and field combination is used for the background, PSF, and flux ratios
        of a single simulated image.

        Returns:
            A dictionary of the simulated TileCatalog, and (batch_size, bands, height, width)
            tensors for images and background.
        """
        rcfs, rcf_indices = self.get_random_rcf(self.image_prior.batch_size)
        with torch.no_grad():
            tile_catalog = self.image_prior.sample_prior(rcfs)
            images, background, deconv = self.simulate_image(tile_catalog, rcf_indices)
            return {
                "tile_catalog": tile_catalog.to_dict(),
                "images": images,
                "background": background,
                "deconvolution": deconv,
            }

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.fix_validation_set:
            valid: List[Dict[str, Tensor]] = []  # type: ignore
            for _ in tqdm(range(self.valid_n_batches), desc="Generating fixed validation set"):
                valid.append(self.get_batch())
            num_workers = 0
        else:
            valid = self
            num_workers = self.num_workers
        return DataLoader(valid, batch_size=None, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)


class CachedSimulatedDataset(pl.LightningDataModule, Dataset):
    def __init__(
        self,
        train_n_batches: int,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        file_prefix: str,
        val_split_file_idxs: List[int],
        test_split_file_idxs: List[int],
    ):
        super().__init__()

        self.train_n_batches = train_n_batches
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cached_data_path = cached_data_path
        self.file_prefix = file_prefix
        self.val_split_file_idxs = val_split_file_idxs or []
        self.test_split_file_idxs = test_split_file_idxs or []

        self.data: List[FileDatum] = []
        self.valid: List[FileDatum] = []
        self.test: List[FileDatum] = []

        # assume cached image files exist, read from disk
        for filename in os.listdir(self.cached_data_path):
            if not filename.endswith(".pt"):
                continue

            file_idx = int(filename.split("_")[-1].split(".")[0])
            if file_idx in self.val_split_file_idxs or file_idx in self.test_split_file_idxs:
                continue
            if filename.startswith(self.file_prefix) and filename.endswith(".pt"):
                self.data += self.read_file(f"{self.cached_data_path}/{filename}")

        # fix validation set
        for idx in self.val_split_file_idxs:
            filename = f"{self.file_prefix}_{idx}.pt"
            self.valid += self.read_file(f"{self.cached_data_path}/{filename}")

        # fix test set
        for idx in self.test_split_file_idxs:
            filename = f"{self.file_prefix}_{idx}.pt"
            self.test += self.read_file(f"{self.cached_data_path}/{filename}")

    def read_file(self, filename: str) -> List[FileDatum]:
        with open(filename, "rb") as f:
            return torch.load(f)

    def __len__(self):
        return self.train_n_batches * self.batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def train_dataloader(self):
        assert self.data, "No cached train data loaded; run `generate.py` first"
        assert len(self.data) >= self.train_n_batches * self.batch_size, (
            f"Insufficient cached data loaded; "
            f"need at least {self.train_n_batches * self.batch_size} "
            f"but only have {len(self.data)}. Re-run `generate.py` with "
            f"different generation `train_n_batches` and/or `batch_size`."
        )
        return DataLoader(
            self, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        assert self.valid, "No cached validation data found; run `generate.py` first"
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        assert self.test, "No cached test data found; run `generate.py` first"
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
