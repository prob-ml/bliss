import math
import os
import random
import warnings
from typing import List, TypedDict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from bliss.catalog import FullCatalog, TileCatalog

# prevent pytorch_lightning warning for num_workers = 2 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)
# an IterableDataset isn't supposed to have a __len__ method
warnings.filterwarnings("ignore", ".*Total length of .* across ranks is zero.*", UserWarning)


FileDatum = TypedDict(
    "FileDatum",
    {
        "tile_catalog": TileCatalog,
        "images": torch.Tensor,
        "background": torch.Tensor,
        "psf_params": torch.Tensor,
    },
)


class FullCatalogToTileTransform(torch.nn.Module):
    def __init__(self, tile_slen, max_sources):
        super().__init__()
        self.tile_slen = tile_slen
        self.max_sources = max_sources

    def __call__(self, ex):
        h_pixels, w_pixels = ex["images"].shape[1:]
        full_cat = FullCatalog(h_pixels, w_pixels, ex["full_catalog"])
        tile_cat = full_cat.to_tile_catalog(self.tile_slen, self.max_sources).data
        d = {k: v.squeeze(0) for k, v in tile_cat.items()}
        ex["tile_catalog"] = d
        del ex["full_catalog"]

        return ex


class MyIterableDataset(IterableDataset):
    def __init__(self, file_paths, shuffle=False, transform=None):
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.transform = transform

    def get_stream(self, files):
        for file_path in files:
            examples = torch.load(file_path)

            # each training worker also shuffles the examples within each file
            if self.shuffle:
                random.shuffle(examples)

            for ex in examples:
                if self.transform is not None:
                    ex = self.transform(ex)
                yield ex

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # shuffle files use for training each epoch
        files = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(files)

        if worker_info is None:  # single-process data loading
            files_subset = files
        else:  # in a worker process
            # split files evenly amongst workers
            per_worker = int(math.ceil(len(files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            files_subset = files[worker_id * per_worker : (worker_id + 1) * per_worker]

        return iter(self.get_stream(files_subset))


class CachedSimulatedDataset(pl.LightningDataModule):
    def __init__(
        self,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
        nontrain_transforms: List,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.nontrain_transforms = nontrain_transforms

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
        assert self.file_paths[self.slices[0]], "No cached data found"
        transform = transforms.Compose(self.train_transforms)
        my_dataset = MyIterableDataset(
            self.file_paths[self.slices[0]], transform=transform, shuffle=True
        )

        return DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=random.seed,
        )

    def _get_nontrain_dataloader(self, file_paths_subset):
        assert file_paths_subset, "No cached data found"
        transform = transforms.Compose(self.nontrain_transforms)
        my_dataset = MyIterableDataset(file_paths_subset, transform=transform)
        return DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=random.seed,
        )

    def val_dataloader(self):
        return self._get_nontrain_dataloader(self.file_paths[self.slices[1]])

    def test_dataloader(self):
        return self._get_nontrain_dataloader(self.file_paths[self.slices[2]])

    def predict_dataloader(self):
        return self._get_nontrain_dataloader(self.file_paths)
