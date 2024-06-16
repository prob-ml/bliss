import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from bliss.align import align
from bliss.catalog import FullCatalog, TileCatalog
from bliss.simulator.decoder import ImageDecoder
from bliss.simulator.prior import CatalogPrior
from bliss.surveys.survey import Survey

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class SimulatedDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self,
        survey: Survey,
        prior: CatalogPrior,
        n_batches: int,
        use_coaddition: bool = False,
        coadd_depth: int = 1,
        num_workers: int = 0,
        valid_n_batches: Optional[int] = None,
        fix_validation_set: bool = False,
    ):
        super().__init__()

        self.survey = survey
        survey.prepare_data()

        self.catalog_prior = prior
        self.background = self.survey.background
        assert self.catalog_prior is not None, "Survey prior cannot be None."
        assert self.background is not None, "Survey background cannot be None."
        self.catalog_prior.requires_grad_(False)
        self.background.requires_grad_(False)

        assert survey.psf is not None, "Survey psf cannot be None."
        assert survey.pixel_shift is not None, "Survey pixel_shift cannot be None."
        assert (
            survey.flux_calibration_dict is not None
        ), "Survey flux_calibration_dict cannot be None."
        self.image_decoder = ImageDecoder(
            psf=survey.psf,
            bands=survey.BANDS,
            pixel_shift=survey.pixel_shift,
            flux_calibration_dict=survey.flux_calibration_dict,
            ref_band=prior.reference_band,
        )

        self.n_batches = n_batches
        self.batch_size = self.catalog_prior.batch_size
        self.use_coaddition = use_coaddition
        self.coadd_depth = coadd_depth
        self.num_workers = num_workers
        self.fix_validation_set = fix_validation_set
        self.valid_n_batches = n_batches if valid_n_batches is None else valid_n_batches

        self.image_ids = self.survey.image_ids()

    def randomized_image_ids(self, num_samples=1) -> Tuple[List[Any], np.ndarray]:
        """Get random image_id from loaded params.

        Args:
            num_samples (int, optional): number of random samples to get. Defaults to 1.

        Returns:
            Tuple[List[Any], np.ndarray]: tuple of image_ids and their corresponding
                `self.image_ids` indices.
        """
        n = np.random.randint(len(self.image_ids), size=(num_samples,), dtype=int)
        # reorder self.image_ids to match the order of the sampled indices
        return [self.image_ids[i] for i in n], n

    def apply_noise(self, images_mean):
        images_mean = torch.clamp(images_mean, min=1e-6)
        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean
        return images

    def coadd_images(self, images):
        batch_size = images.shape[0]
        assert self.coadd_depth > 1, "Coadd depth must be > 1 to use coaddition."
        coadded_images = np.zeros((batch_size, *images.shape[-3:]))  # 4D
        for b in range(batch_size):
            coadded_images[b] = self.survey.coadd_images(images[b])
        return torch.from_numpy(coadded_images).float()

    def align_images(self, images, wcs_batch):
        """Align images to the reference depth and band."""
        batch_size = images.shape[0]
        for b in range(batch_size):
            aligned_image = align(
                images[b].numpy(),
                wcs_list=wcs_batch[b],
                ref_depth=0,
                ref_band=self.catalog_prior.reference_band,
            )
            images[b] = torch.from_numpy(aligned_image)
        return images

    def simulate_images(
        self, tile_catalog: TileCatalog, image_ids, image_id_indices
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Simulate a batch of images.

        Args:
            tile_catalog (TileCatalog): The TileCatalog to render.
            image_ids: List of image_ids to render, in `image_id_indices` order.
            image_id_indices: Indices in self.image_ids to sample from.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: tuple of images, backgrounds, and psf parameters.
        """
        images, _, psf_params, wcs_batch = self.image_decoder.render_images(
            tile_catalog, image_ids, self.coadd_depth
        )
        images = self.align_images(images, wcs_batch)
        if self.use_coaddition:
            images = self.coadd_images(images)

        background = self.background.sample(images.shape, image_id_indices=image_id_indices)
        images += background

        images = self.apply_noise(images)
        return images, background, psf_params

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
        assert self.catalog_prior is not None, "Survey prior cannot be None."

        image_ids, image_id_indices = self.randomized_image_ids(self.catalog_prior.batch_size)
        with torch.no_grad():
            tile_catalog = self.catalog_prior.sample()
            images, background, psf_params = self.simulate_images(
                tile_catalog, image_ids, image_id_indices
            )
            return {
                "tile_catalog": tile_catalog,
                "images": images,
                "background": background,
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


FileDatum = TypedDict(
    "FileDatum",
    {
        "tile_catalog": TileCatalog,
        "images": torch.Tensor,
        "background": torch.Tensor,
        "psf_params": torch.Tensor,
    },
)


class MyIterableDataset(IterableDataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_stream(self, files):
        # TODO: shuffle files
        for file_path in files:
            examples = torch.load(file_path)
            # TODO: randomly sort examples
            for ex in examples:
                if "full_catalog" in ex:
                    full_cat = FullCatalog(112, 112, ex["full_catalog"])
                    tile_cat = full_cat.to_tile_catalog(2, 6).data
                    d = {k: v.squeeze(0) for k, v in tile_cat.items()}
                    ex["tile_catalog"] = d
                    del ex["full_catalog"]
                yield ex

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            files = self.file_paths
        else:  # in a worker process
            # Split workload
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            files = self.file_paths[worker_id * per_worker : (worker_id + 1) * per_worker]

        return iter(self.get_stream(files))


class CachedSimulatedDataset(pl.LightningDataModule):
    def __init__(
        self,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        file_names = [f for f in os.listdir(cached_data_path) if f.endswith(".pt")]
        self.file_paths = [os.path.join(cached_data_path, f) for f in file_names]

        # parse slices from percentages to indices
        self.slices = self.parse_slices(splits, len(self.file_paths))

    def _percent_to_idx(self, x, length):
        """Converts string in percent to an integer index."""
        return int(float(x.strip()) / 100 * length) if x.strip() else None

    def parse_slices(self, splits: str, length: int):
        slices = [slice(0, 0) for _ in range(3)]  # default to empty slice for each split
        for i, data_split in enumerate(splits.split("/")):
            # map "start_percent:stop_percent" to slice(start_idx, stop_idx)
            slices[i] = slice(*(self._percent_to_idx(val, length) for val in data_split.split(":")))
        return slices

    def train_dataloader(self):
        assert self.file_paths[self.slices[0]], "No cached validation data found"
        return DataLoader(
            MyIterableDataset(self.file_paths[self.slices[0]]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        assert self.file_paths[self.slices[1]], "No cached validation data found"
        return DataLoader(
            MyIterableDataset(self.file_paths[self.slices[1]]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        assert self.file_paths[self.slices[2]], "No cached test data found"
        return DataLoader(
            MyIterableDataset(self.file_paths[self.slices[2]]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        assert self.file_paths, "No cached data found"
        return DataLoader(self.file_paths, batch_size=self.batch_size, num_workers=self.num_workers)
