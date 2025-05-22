import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from astropy.io import fits
from torch import distributed
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

CACHED_DATA_PATH = (
    "/nfs/turbo/lsa-regier/scratch/gapatron/desdr-server.ncsa.illinois.edu/despublic/dr2_tiles"
)
DES_BANDS = ("g", "r", "i", "z")


class DESSampler(Sampler):
    def __init__(self, dataset: Dataset) -> None:
        assert isinstance(dataset, Dataset), "dataset should be Dataset"
        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset.get_indices())


class DistributedDESSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        assert isinstance(dataset, Dataset), "dataset should be Dataset"
        assert not shuffle, "you should not use shuffle"
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self):
        num_big_images = len(self.dataset.directories)
        indices = list(range(num_big_images))
        rank_indices = indices[self.rank :: self.num_replicas]

        # Convert big image indices to tile indices
        tile_indices = []
        for big_image_idx in rank_indices:
            start_idx = big_image_idx * self.dataset.tiles_per_img
            end_idx = start_idx + self.dataset.tiles_per_img
            tile_indices.extend(range(start_idx, end_idx))

        return iter(tile_indices)


class DESDataset(Dataset):
    def __init__(self, cached_data_path: str, tiles_per_img: int) -> None:
        super().__init__()
        self.directory_paths = cached_data_path
        self.directories = [d for d in os.listdir(self.directory_paths) if d.startswith("DES")]
        self.buffer = None
        self.tiles_per_img = tiles_per_img

    def _build_image(self, directory_path):
        dir_files = {
            band: [
                f for f in os.listdir(f"{directory_path}") if f.endswith(f"_{band}.fits.fz")
            ][0]
            for band in DES_BANDS
        }
        image_bands = []
        for band in DES_BANDS:
            band_filepath = f"{directory_path}/{dir_files[band]}"
            with fits.open(band_filepath) as f:
                # Data seems to be on HDU 1, not 0.
                hud = torch.from_numpy(f[1].data)
            image_bands.append(hud.data.unsqueeze(0))

        return torch.cat(image_bands, axis=0)

    def __getitem__(self, idx):
        # Calculate the directory index and tile index for the given index
        dir_idx = idx // self.tiles_per_img
        tile_idx = idx % self.tiles_per_img

        if self.buffer is None or dir_idx != self.buffer[0]:
            des_dir_path = Path(self.directory_paths, self.directories[dir_idx])
            # Resulting items are (4, 10000, 10000)
            image_item = self._build_image(des_dir_path)
            # Resulting items are (4, 8, 8, 1280, 1280)
            item = image_item.unfold(dimension=1, size=1280, step=1235).unfold(
                dimension=2, size=1280, step=1235
            )
            # Finally obtain (4, 64, 1280, 1280)
            item = item.reshape(4, -1, 1280, 1280)
            self.buffer = (dir_idx, item)

        # Return selected tile index.
        return self.buffer[1][:, tile_idx]

    def __len__(self):
        return len(self.directories) * self.tiles_per_img


class CachedDESModule(pl.LightningDataModule):
    def __init__(
        self,
        cached_data_path: str,
        tiles_per_img: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cached_data_path = Path(cached_data_path)
        self.tiles_per_img = tiles_per_img
        self.predict_dataset = DESDataset(self.cached_data_path, self.tiles_per_img)

    def _get_dataloader(self, dataset):
        distributed_is_used = distributed.is_available() and distributed.is_initialized()
        if distributed_is_used:
            sampler = DistributedDESSampler(
                dataset,
                num_replicas=distributed.get_world_size(),
                rank=distributed.get_rank(),
                shuffle=False,
                seed=42,
            )
        else:
            sampler = DESSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
        )

    def predict_dataloader(self):
        return self._get_dataloader(self.predict_dataset)
