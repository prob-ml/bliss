import logging
import math
import os
import pathlib
import random
import re
import warnings
from copy import copy
from typing import List

import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms

from bliss.catalog import FullCatalog, TileCatalog
from bliss.global_env import GlobalEnv

# prevent pytorch_lightning warning for num_workers = 2 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)
# an IterableDataset isn't supposed to have a __len__ method
warnings.filterwarnings("ignore", ".*Total length of .* across ranks is zero.*", UserWarning)


class FullCatalogToTileTransform(torch.nn.Module):
    def __init__(self, tile_slen, max_sources):
        super().__init__()
        self.tile_slen = tile_slen
        self.max_sources = max_sources

    def __call__(self, datum_in):
        datum_out = {k: v for k, v in datum_in.items() if k != "full_catalog"}

        h_pixels, w_pixels = datum_in["images"].shape[1:]
        d1 = {k: v.unsqueeze(0) for k, v in datum_in["full_catalog"].items()}
        full_cat = FullCatalog(h_pixels, w_pixels, d1)
        tile_cat = full_cat.to_tile_catalog(self.tile_slen, self.max_sources).data
        d2 = {k: v.squeeze(0) for k, v in tile_cat.items()}
        datum_out["tile_catalog"] = d2

        return datum_out


class OneBandTransform(torch.nn.Module):
    def __init__(self, band_idx):
        super().__init__()
        self.band_idx = band_idx

    def __call__(self, datum_in):
        datum_out = {
            "images": datum_in["images"][self.band_idx : self.band_idx + 1],
            "psf_params": datum_in["psf_params"][self.band_idx : self.band_idx + 1],
        }

        for cat_name in ("tile_catalog", "full_catalog"):
            if cat_name not in datum_in:
                continue
            cat = copy(datum_in[cat_name])
            for k, v in cat.items():
                if k.endswith("fluxes"):
                    cat[k] = v[..., self.band_idx : self.band_idx + 1]
            datum_out[cat_name] = cat

        return datum_out


class FluxFilterTransform(torch.nn.Module):
    def __init__(self, reference_band, min_flux):
        super().__init__()
        self.reference_band = reference_band
        self.min_flux = min_flux

    def __call__(self, datum_in):
        datum_out = copy(datum_in)

        d1 = {k: v.unsqueeze(0) for k, v in datum_in["tile_catalog"].items()}
        target_cat = TileCatalog(d1)
        target_cat = target_cat.filter_by_flux(
            min_flux=self.min_flux,
            band=self.reference_band,
        )
        d2 = {k: v.squeeze(0) for k, v in target_cat.items()}
        datum_out["tile_catalog"] = d2

        return datum_out


class ChunkingSampler(Sampler):
    def __init__(self, dataset: Dataset) -> None:
        # please don't pass dataset to the following __init__()
        # according to https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
        # the parameter `data_source` has been deprecated
        # make sure your pytorch version is greater than 2.2.0
        super().__init__()
        assert isinstance(dataset, ChunkingDataset), "dataset should be ChunkingDataset"
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset.get_chunked_indices())


# note that for this DistributedSampler, we don't need to call `set_epoch()`
class DistributedChunkingSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        assert isinstance(dataset, ChunkingDataset), "dataset should be ChunkingDataset"
        assert not shuffle, "you should not use shuffle"
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self):
        pre_indices = super().__iter__()
        chunked_indices = self.dataset.get_chunked_indices()

        return iter([chunked_indices[i] for i in pre_indices])


class ChunkingDataset(Dataset):
    def __init__(self, file_paths, shuffle=False, transform=None) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.transform = transform

        self.accumulated_file_sizes = torch.zeros(len(self.file_paths), dtype=torch.int64)
        for i, file_path in enumerate(self.file_paths):
            file_size_match = re.search(r"size_(\d+)", file_path)
            if file_size_match:
                cached_data_len = int(file_size_match.group(1))
            else:
                if i == 0:
                    # we assume in your data file, you use a list to hold the data points
                    # the postfix '_size_<chunk size>' is the length of this list
                    logger = logging.getLogger("ChunkingDataset")
                    warning_msg = (
                        "WARNING: add postfix '_size_<chunk size>' to file name; "
                        "otherwise it'll be very slow \n"
                        "for instance, a valid file name can be 'cached_data_size_500.pt' "
                        "(for more details, please see cached_dataset.py)\n"
                    )
                    logger.warning(warning_msg)
                with open(file_path, "rb") as f:
                    cached_data_len = len(torch.load(f))

            if i == 0:
                self.accumulated_file_sizes[i] = cached_data_len
            else:
                self.accumulated_file_sizes[i] = (
                    self.accumulated_file_sizes[i - 1] + cached_data_len
                )

        self.buffered_file_index = None
        self.buffered_data = None

    def __len__(self):
        return self.accumulated_file_sizes[-1].item()

    def __getitem__(self, index):
        converted_index = (self.accumulated_file_sizes <= index).sum().item()
        converted_sub_index = (index - self.accumulated_file_sizes[converted_index]).item()
        if self.buffered_file_index != converted_index:
            self.buffered_file_index = converted_index
            with open(self.file_paths[converted_index], "rb") as f:
                self.buffered_data = torch.load(f)
        output_data = self.buffered_data[converted_sub_index]
        return self.transform(output_data)

    def get_chunked_indices(self):
        accumulated_file_sizes_list = self.accumulated_file_sizes.tolist()

        output_list = []
        if self.shuffle:
            epoch_seed = GlobalEnv.seed_in_this_program + GlobalEnv.current_encoder_epoch
            logger = logging.getLogger("ChunkingDataset")
            logger.info(
                "INFO: seed is %d; current epoch is %d; epoch_seed is set to %d",
                GlobalEnv.seed_in_this_program,
                GlobalEnv.current_encoder_epoch,
                epoch_seed,
            )
            right_shift_list = [0, *accumulated_file_sizes_list[:-1]]
            for start, end in zip(right_shift_list, accumulated_file_sizes_list):
                output_list.append(random.Random(epoch_seed).sample(range(start, end), end - start))
            random.Random(epoch_seed).shuffle(output_list)
            # flatten the list
            return sum(output_list, [])

        return list(range(0, len(self)))


class CachedSimulatedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
        nontrain_transforms: List,
        subset_fraction: float = None,
    ):
        super().__init__()

        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cached_data_path = pathlib.Path(cached_data_path)
        self.train_transforms = train_transforms
        self.nontrain_transforms = nontrain_transforms
        self.subset_fraction = subset_fraction

        self.file_paths = None
        self.slices = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str) -> None:  # noqa: WPS324
        if self.file_paths is None or self.slices is None:
            self._load_file_paths_and_slices()

        if stage == "fit":
            self.train_dataset = self._get_dataset(
                self.file_paths[self.slices[0]], self.train_transforms, shuffle=True
            )

            self.val_dataset = self._get_dataset(
                self.file_paths[self.slices[1]], self.nontrain_transforms
            )
            return None

        if stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self._get_dataset(
                    self.file_paths[self.slices[1]], self.nontrain_transforms
                )
            return None

        if stage == "test":
            self.test_dataset = self._get_dataset(
                self.file_paths[self.slices[2]], self.nontrain_transforms
            )
            return None

        if stage == "predict":
            self.predict_dataset = self._get_dataset(self.file_paths, self.nontrain_transforms)
            return None

        raise RuntimeError(f"setup skips stage {stage}")

    def _load_file_paths_and_slices(self):
        file_names = [f for f in os.listdir(str(self.cached_data_path)) if f.endswith(".pt")]
        if self.subset_fraction:
            file_names = file_names[: math.ceil(len(file_names) * self.subset_fraction)]
        self.file_paths = [os.path.join(str(self.cached_data_path), f) for f in file_names]

        # parse slices from percentages to indices
        self.slices = self.parse_slices(self.splits, len(self.file_paths))

    def _percent_to_idx(self, x, length):
        """Converts string in percent to an integer index."""
        return int(float(x.strip()) / 100 * length) if x.strip() else None

    def parse_slices(self, splits: str, length: int):
        slices = [slice(0, 0) for _ in range(3)]  # default to empty slice for each split
        for i, data_split in enumerate(splits.split("/")):
            # map "start_percent:stop_percent" to slice(start_idx, stop_idx)
            slices[i] = slice(*(self._percent_to_idx(val, length) for val in data_split.split(":")))
        return slices

    def _get_dataset(self, sub_file_paths, defined_transforms, shuffle: bool = False):
        assert sub_file_paths, "No cached data found"
        transform = transforms.Compose(defined_transforms)
        return ChunkingDataset(sub_file_paths, shuffle=shuffle, transform=transform)

    def _get_dataloader(self, my_dataset):
        distributed_is_used = dist.is_available() and dist.is_initialized()
        sampler_type = DistributedChunkingSampler if distributed_is_used else ChunkingSampler
        return DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler_type(my_dataset),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self._get_dataloader(self.predict_dataset)
