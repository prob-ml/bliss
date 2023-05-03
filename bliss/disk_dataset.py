import warnings
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.simulator.background import ConstantBackground, SimulatedSDSSBackground
from bliss.simulator.decoder import ImageDecoder
from bliss.simulator.prior import ImagePrior

import pickle
import os

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class DiskDataset(pl.LightningDataModule, Dataset):
    # stores details of the written image files - { filename: string, data }
    data_files = []

    def __init__(
        self,
        prior: ImagePrior,
        decoder: ImageDecoder,
        background: Union[ConstantBackground, SimulatedSDSSBackground],
        n_batches: int,
        data_path: str,
        num_workers: int = 0,
        disk_batch_size: int = 5000,
    ):
        super().__init__()

        self.n_batches = n_batches
        self.image_prior = prior
        self.image_decoder = decoder
        self.background = background
        self.image_prior.requires_grad_(False)
        self.background.requires_grad_(False)
        self.num_workers = num_workers

        self.data_path = data_path
        self.disk_batch_size = disk_batch_size
        self.disk_n_batches = -(n_batches * prior.batch_size // -disk_batch_size) # ceil division

        # populate self.data_files with written pkl image files, if any
        for filename in os.listdir(f'{self.data_path}/'):
            if filename.startswith('dataset_') and filename.endswith('.pkl'):
                self.data_files.append({'filename': filename, 'data': None})

        if len(self.data_files) != self.disk_n_batches:
            # if no written image files, render (and write to disk) images
            for disk_batch_idx in range(self.disk_n_batches):
                batch_data = self.get_disk_batch()

                # write `disk_batch_size` images to disk
                data_file = open(f'{self.data_path}/dataset_{disk_batch_idx}.pkl', 'wb')
                self.data_files.append({'filename': data_file.name, 'data': batch_data})
                pickle.dump(batch_data, data_file)
                data_file.close()
        else:
            # else, read images from disk
            for disk_batch_idx in range(self.disk_n_batches):
                self.data_files[disk_batch_idx]['data'] = self.read_batch(disk_batch_idx)

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.
        assert torch.all(images_mean > 1e-8)
        images = torch.sqrt(images_mean) * torch.randn_like(images_mean)
        images += images_mean
        return images

    def simulate_image(self, tile_catalog: TileCatalog) -> Tuple[Tensor, Tensor]:
        images = self.image_decoder.render_images(tile_catalog)
        background = self.background.sample(images.shape)
        images += background
        images = self._apply_noise(images)
        return images, background

    def get_disk_batch(self) -> Dict:
        with torch.no_grad():
            tile_catalog = self.image_prior.sample_prior(self.disk_batch_size)
            images, background = self.simulate_image(tile_catalog)
            return {
                "tile_catalog": tile_catalog.to_dict(),
                "images": images,
                "background": background,
            }
        
    def read_batch(self, disk_batch_idx) -> Dict:
        # read images from disk, by batch
        data_file = open(f'{self.data_path}/{self.data_files[disk_batch_idx]["filename"]}', 'rb')
        
        try:
            batch_data = pickle.load(data_file)
            data_file.close()
            return batch_data
        except EOFError:
            print(f'Error: EOFError while reading {self.data_files[disk_batch_idx]["filename"]}')
            raise EOFError
        
    def __len__(self):
        return self.disk_n_batches * self.disk_batch_size
        
    def __getitem__(self, idx):
        # convert idx to disk_batch_idx and idx_in_batch
        disk_batch_idx = idx // self.disk_batch_size
        idx_in_batch = idx % self.disk_batch_size

        # return the idx_in_batch-th entry of the disk_batch_idx-th batch
        tile_catalog_dict = {}
        for key in self.data_files[disk_batch_idx]['data']['tile_catalog']:
            tile_catalog_dict[key] = self.data_files[disk_batch_idx]['data']['tile_catalog'][key][idx_in_batch]
        image = self.data_files[disk_batch_idx]['data']['images'][idx_in_batch]
        background = self.data_files[disk_batch_idx]['data']['background'][idx_in_batch]
        return { "tile_catalog": tile_catalog_dict, "images": image, "background": background }

    def train_dataloader(self):
        return DataLoader(self, batch_size=self.image_prior.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        valid = self
        num_workers = self.num_workers
        return DataLoader(valid, batch_size=self.image_prior.batch_size, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self, batch_size=self.image_prior.batch_size, num_workers=self.num_workers)
