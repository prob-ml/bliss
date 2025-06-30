from bliss.cached_dataset import CachedSimulatedDataModule
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
from typing import List
from torch.utils.data import Dataset, Sampler


class GalaxyClusterCachedSimulatedDataset(Dataset):
    def __init__(
            self, 
            sub_file_paths,
            transform=None,
                 ):
        super().__init__()
        self.sub_file_paths   = [subtile_filepath for subtile_filepath in sub_file_paths if ("_subtile_" in subtile_filepath and subtile_filepath.endswith(".pt"))]
        self.transform        = transform
        

    def _load_tile(self, path):
        with open(path, "rb") as f:
            data = torch.load(f, map_location=torch.device('cpu'))
        return self.transform(data) if self.transform else data

    def __len__(self):
        return len(self.sub_file_paths)

    def __getitem__(self, idx):

        subtile_data = self._load_tile(self.sub_file_paths[idx])
        if subtile_data is None:
            raise ValueError(f"Data at index {idx} is None. Check the file path: {self.sub_file_paths[idx]}")
        batch = {
            'images': subtile_data['images'],
            'tile_catalog': {
                'membership': subtile_data['tile_catalog']['membership'].squeeze(-1),
                'locs': torch.zeros((4,4,1,2), dtype=torch.float32),  # placeholder for locs
            }
        }
        return batch



class GalaxyClusterSampler(Sampler):
    """Sampler for galaxy cluster datasets that ensures each sample is unique."""
    def __init__(self, dataset):
        self.dataset = dataset


    def __iter__(self):
        for i in range(0, len(self.dataset)):
            yield i

    def __len__(self):
        return len(self.dataset)
    
class DistributedGalaxyClusterSampler(DistributedSampler):
    """Distributed sampler for galaxy cluster datasets."""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.dataset = dataset

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices)

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
        )

    def _get_dataloader(self, my_dataset):
        distributed_is_used = dist.is_available() and dist.is_initialized()
        #sampler_type = DistributedChunkingSampler if distributed_is_used else BufferSampler
        if distributed_is_used:
            print("Using distributed sampler")
            sampler_type = DistributedGalaxyClusterSampler
        else:
            sampler_type = GalaxyClusterSampler
        return DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler_type(my_dataset),
            #prefetch_factor=2,
        )