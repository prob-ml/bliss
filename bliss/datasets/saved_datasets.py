"""Datasets actually used to train models based on cached galaxy iamges."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from bliss.catalog import FullCatalog
from bliss.datasets.io import load_dataset_npz


class SavedGalsimBlends(Dataset):
    def __init__(
        self,
        dataset_file: str,
        slen: int = 40,
        tile_slen: int = 4,
        keep_padding: bool = False,
    ) -> None:
        super().__init__()
        ds: dict[str, Tensor] = load_dataset_npz(dataset_file)

        self.images = ds.pop("images")
        self.epoch_size = len(self.images)

        # don't need for training
        ds.pop("centered_sources")
        ds.pop("uncentered_sources")
        ds.pop("noiseless")

        # avoid large memory usage if we don't need padding.
        if not keep_padding:
            ds.pop("paddings")
            self.paddings = torch.tensor([0]).float()
        else:
            self.paddings = ds.pop("paddings")
        self.keep_padding = keep_padding

        full_catalog = FullCatalog(slen, slen, ds)
        tile_catalogs = full_catalog.to_tile_params(tile_slen, ignore_extra_sources=True)
        self.tile_params = tile_catalogs.to_dict()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "paddings": self.paddings[index] if self.keep_padding else self.paddings,
            **tile_params_ii,
        }


class SavedIndividualGalaxies(Dataset):
    def __init__(self, dataset_file: str) -> None:
        super().__init__()
        ds: dict[str, Tensor] = load_dataset_npz(dataset_file)

        self.images = ds.pop("images")
        self.epoch_size = len(self.images)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        return {"images": self.images[index]}
