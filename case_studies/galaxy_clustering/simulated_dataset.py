import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset, IterableDataset

from bliss.catalog import FullCatalog
from bliss.simulator.simulated_dataset import CachedSimulatedDataset, FileDatum

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class GalaxyClusterCachedSimulatedDataset(CachedSimulatedDataset):
    def __init__(
        self,
        splits: str,
        batch_size: int,
        num_workers: int,
        cached_data_path: str,
        file_prefix: str,
    ):
        super().__init__(splits, batch_size, num_workers, cached_data_path, file_prefix)