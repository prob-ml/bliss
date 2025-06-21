from bliss.cached_dataset import CachedSimulatedDataModule
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
from typing import List
from torch.utils.data import Dataset, Sampler


class GalaxyClusterCachedSimulatedDataset(Dataset):
    def __init__(self, sub_file_paths, transform=None,
                 buffer_size=10):
        super().__init__()
        self.sub_file_paths   = sub_file_paths
        self.transform        = transform
        self.buffer_size      = buffer_size
        self._make_random_buffers()
        self._cur_buf_idx = -1
        self._buf_data   = None
        # When unfolding a tile with subtile size of 2560 and
        # step size of 1024, we can sample 16 times per tile.
        self.num_samples_per_tile = 64


    def _make_random_buffers(self, epoch_seed=None):
        """
        Shuffle tiles > slice into random buffers.
        Saves
            self.buffer_groups: list of lists, each containing tile indices
            self.tile2buf: dict mapping tile_id to buffer_id
        """

        perm = torch.randperm(len(self.sub_file_paths)).tolist()
        self.buffer_groups = [
            perm[i : i + self.buffer_size]
            for i in range(0, len(perm), self.buffer_size)
        ]
        # Reverse map: “tile_id → buffer_id”
        self.tile2buf = {
            tile_id: buf_id
            for buf_id, group in enumerate(self.buffer_groups)
            for tile_id in group
        }
        self.tile2local = {
            tile_id: local
            for buf_id, group in enumerate(self.buffer_groups)
            for local, tile_id in enumerate(group)
        }

    def _load_tile(self, path):
        with open(path, "rb") as f:
            data = torch.load(f, map_location=torch.device('cpu'))
        return self.transform(data) if self.transform else data

    def _ensure_buffer(self, tile_idx):
        """
        Ensure that the buffer containing tile_idx is loaded into memory.
        This is called by __getitem__ to load the appropriate buffer
        for the requested tile.

        This version is memory-efficient: it only loads the buffer
        if it is not already loaded, and it does not keep any
        intermediate data in memory after use.
        """
        buf_idx = self.tile2buf[tile_idx]
        if buf_idx == self._cur_buf_idx:
            return
        tile_ids = self.buffer_groups[buf_idx]
        paths = [self.sub_file_paths[i] for i in tile_ids]
        B = len(tile_ids)

        tiles = [self._load_tile(p)[0] for p in paths]  
        img_buffer = torch.stack([t['images']                       for t in tiles])
        mem_buffer = torch.stack([t['tile_catalog']['membership']   for t in tiles])

        self._buf_data    = {
            'images': img_buffer.unfold(2, 3072, 768).unfold(3, 3072, 768).permute(0,2,3,1,4,5).contiguous().reshape(img_buffer.size(0), -1,  img_buffer.size(1), 3024, 3024),
            'tile_catalog': mem_buffer.unfold(1, 4, 1).unfold(2, 4, 1).permute(0, 1, 2, 5, 6, 3, 4).contiguous().reshape(mem_buffer.size(0), -1, 11, 11, 1, 1),
        }
        self._cur_buf_idx = buf_idx

    def __len__(self):
        return len(self.sub_file_paths) * self.num_samples_per_tile

    def __getitem__(self, idx):
        tile_idx, sample_idx = divmod(idx, self.num_samples_per_tile)
        # Make sure the right buffer is in memory
        self._ensure_buffer(tile_idx)

        local_idx = self.tile2local[tile_idx]
        subimage_tile = self._buf_data['images'][local_idx, sample_idx]
        subcat_tile = self._buf_data['tile_catalog'][local_idx, sample_idx]
        batch = {
            'images': subimage_tile,
            'tile_catalog': {
                'membership': subcat_tile,
                'locs': torch.zeros((2,2,1,2), dtype=torch.float32),  # placeholder for locs
            }
        }
        return batch



class BufferSampler(Sampler):
    """
    Yields sample indices so that a worker consumes all samples from buffer-0,
    then buffer-1, …  Buffers themselves are randomised every epoch.
    """

    def __init__(self, dataset, shuffle=True):
        self.dataset          = dataset
        self.samples_per_tile = self.dataset.num_samples_per_tile
        self.buffer_size      = self.dataset.buffer_size
        self.shuffle          = shuffle
        self.epoch            = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):

        # Yield indices randomly from buffers.
        for buf in self.dataset.buffer_groups:
            tile_indices = []
            for tile_id in buf:
                base = tile_id * self.samples_per_tile
                tile_indices.extend(range(base, base + self.samples_per_tile))
            if self.shuffle:
                tile_indices = [tile_indices[index] for index in torch.randperm(len(tile_indices)).tolist()]
            for idx in tile_indices:
                yield idx

    def __len__(self):
        return len(self.dataset)


class DistributedBufferSampler(DistributedSampler):
    def __init__(self, dataset, shuffle=True, seed=0, drop_last=False,
                 num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

        self.buffer_groups = dataset.buffer_groups
        if len(self.buffer_groups) == 1:
            self.buffer_groups = self.buffer_groups * dist.get_world_size()
        self.samples_per_tile = dataset.num_samples_per_tile

    def __iter__(self):
        buffers = self.buffer_groups

        assigned_buffers = buffers[self.rank::self.num_replicas]

        print(f"[Rank {self.rank}] Assigned buffers: {assigned_buffers}")

        all_indices = []
        for buf in assigned_buffers:
            tile_indices = []
            for tile_id in buf:
                base = tile_id * self.samples_per_tile
                tile_indices.extend(range(base, base + self.samples_per_tile))

            if self.shuffle:
                # without self.epoch, the shuffle of each epoch would be the same
                buf_seed = self.seed + hash(tuple(buf)) % (2**31)
                local_rng = random.Random(buf_seed)
                local_rng.shuffle(tile_indices)

            all_indices.extend(tile_indices)

        return iter(all_indices)


class GalaxyClusterCachedSimulatedDataModule(CachedSimulatedDataModule):
    """Data module for cached simulated galaxy cluster datasets."""

    def __init__(
        self,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        train_transforms: List,
        nontrain_transforms: List,
        subset_fraction: float = None,
        buffer_size: int = 10,
    ):
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            cached_data_path=cached_data_path,
            train_transforms=train_transforms,
            nontrain_transforms=nontrain_transforms,
            subset_fraction=subset_fraction,
        )
        self.buffer_size = buffer_size

    @property
    def dataset_name(self):
        return "galaxy_cluster_cached_simulated"

    @property
    def dataset_version(self):
        return "DES DR1"

    @property
    def dataset_description(self):
        return "Cached simulated galaxy cluster dataset for encoder training."

    def _get_dataset(self, sub_file_paths, defined_transforms, shuffle=False):
        assert sub_file_paths, "No sub-file paths provided for dataset."
        transform = transforms.Compose(defined_transforms) if defined_transforms else None
        return GalaxyClusterCachedSimulatedDataset(
            sub_file_paths=sub_file_paths,
            transform=transform,
            buffer_size=self.buffer_size,
        )

    def _get_dataloader(self, my_dataset):
        distributed_is_used = dist.is_available() and dist.is_initialized()
        #sampler_type = DistributedChunkingSampler if distributed_is_used else BufferSampler
        if distributed_is_used:
            print("Using distributed buffer sampler")
            sampler_type = DistributedBufferSampler
        else:
            sampler_type = BufferSampler
        return DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler_type(my_dataset),
        )