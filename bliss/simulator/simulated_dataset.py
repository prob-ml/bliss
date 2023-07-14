import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from skimage.restoration import richardson_lucy
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.generate import FileDatum
from bliss.simulator.decoder import ImageDecoder
from bliss.surveys.survey import Survey

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        survey: Survey,
        n_batches: int,
        num_workers: int = 0,
        valid_n_batches: Optional[int] = None,
        fix_validation_set: bool = False,
    ):
        super().__init__()

        self.survey = survey
        self.image_prior = self.survey.prior
        self.background = self.survey.background
        assert self.image_prior is not None, "Survey prior cannot be None."
        assert self.background is not None, "Survey background cannot be None."
        self.image_prior.requires_grad_(False)
        self.background.requires_grad_(False)

        assert survey.psf is not None, "Survey psf cannot be None."
        assert survey.bands is not None, "Survey bands cannot be None."
        self.image_decoder = ImageDecoder(
            psf=survey.psf, bands=survey.bands, nmgy_to_nelec_dict=survey.nmgy_to_nelec_dict
        )

        self.n_batches = n_batches
        self.batch_size = self.image_prior.batch_size
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

        # list of (run, camcol, field) tuples from config
        self.image_ids = np.array(self.survey.image_ids())

    def randomized_image_ids(self, num_samples=1) -> Tuple[np.ndarray, np.ndarray]:
        """Get random image_id from loaded params.

        Args:
            num_samples (int, optional): number of random samples to get. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of image_ids and their corresponding
                `self.image_ids` indices.
        """
        n = np.random.randint(len(self.image_ids), size=(num_samples,), dtype=int)
        return self.image_ids[n], n

    def _apply_noise(self, images_mean):
        # add noise to images.
        assert torch.all(images_mean > 1e-8)
        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean
        return images

    def simulate_image(
        self, tile_catalog: TileCatalog, image_id_indices
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Simulate a batch of images.

        Args:
            tile_catalog (TileCatalog): The TileCatalog to render.
            image_id_indices: Indices in self.image_ids to sample from.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: tuple of images, backgrounds, deconvolved images,
            and psf parameters
        """
        image_ids = self.image_ids[image_id_indices]
        images, psfs, psf_params = self.image_decoder.render_images(tile_catalog, image_ids)
        assert self.background is not None, "Survey background cannot be None."
        background = self.background.sample(images.shape, image_id_indices=image_id_indices)
        images += background
        images = self._apply_noise(images)
        deconv_images = self.get_deconvolved_images(images, background, psfs)
        return images, background, deconv_images, psf_params

    def get_deconvolved_images(self, images, backgrounds, psfs) -> Tensor:
        """Deconvolve the synthetic images with the psf used to generate them.

        Args:
            images (Tensor): batch of images
            backgrounds (Tensor): batch of backgrounds
            psfs (ndarray): batch of psfs

        Returns:
            Tensor: batch of deconvolved images
        """
        assert self.image_prior is not None, "Survey prior cannot be None."

        deconv_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            for band in range(self.image_prior.n_bands):
                deconv_images[i][band] = self.deconvolve_image(
                    images[i][band], backgrounds[i][band], psfs[i][band]
                )
        return torch.from_numpy(deconv_images)

    def deconvolve_image(self, image, background, psf, pad=10):
        """Deconvolve a single image.

        Args:
            image (Tensor): the image to deconvolve
            background (Tensor): background of the image (used for padding)
            psf (ndarray): the psf used to generate the image
            pad (int): the pad width (in pixels). Defaults to 10.

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
            Dict: A dictionary of the simulated TileCatalog, (batch_size, bands, height, width)
            tensors for images and background, and a (batch_size, 1, 6) tensor for the psf params.
        """
        assert self.image_prior is not None, "Survey prior cannot be None."

        _, image_id_indices = self.randomized_image_ids(self.image_prior.batch_size)
        with torch.no_grad():
            tile_catalog = self.image_prior.sample_prior()
            images, background, deconv, psf_params = self.simulate_image(
                tile_catalog, image_id_indices
            )
            return {
                "tile_catalog": tile_catalog.to_dict(),
                "images": images,
                "background": background,
                "deconvolution": deconv,
                "psf_params": psf_params,
            }

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.fix_validation_set:
            valid: List[Dict[str, Tensor]] = []
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
        bands: List,
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
        self.bands = bands
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
