"""Datasets actually used to train models based on cached galaxy iamges."""

from torch import Tensor
from torch.utils.data import Dataset

from bliss.catalog import FullCatalog
from bliss.datasets.io import load_dataset_npz


class SavedGalsimBlends(Dataset):
    def __init__(self, dataset_file: str, *, slen: int = 50, tile_slen: int = 5) -> None:
        super().__init__()
        ds: dict[str, Tensor] = load_dataset_npz(dataset_file)

        self.images = ds.pop("images")
        self.epoch_size = len(self.images)

        # don't need for training relevant encoders
        ds.pop("centered_sources")
        ds.pop("uncentered_sources")
        ds.pop("noiseless")
        ds.pop("galaxy_params")
        ds.pop("star_fluxes")
        ds.pop("star_bools")
        ds.pop("paddings")

        full_catalog = FullCatalog(slen, slen, ds)
        tile_catalogs = full_catalog.to_tile_params(tile_slen, ignore_extra_sources=True)
        self.tile_params = tile_catalogs.to_dict()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            **tile_params_ii,
        }


class SavedPtiles(Dataset):
    """Currently only used for deblender encoder training."""

    def __init__(self, dataset_file: str) -> None:
        super().__init__()
        ds: dict[str, Tensor] = load_dataset_npz(dataset_file)

        self.images = ds.pop("images")
        self.epoch_size = len(self.images)

        noise = self.images - ds.pop("uncentered_sources") - ds.pop("paddings")
        centered = ds.pop("centered_sources") + noise
        self.centered = centered

        ds.pop("n_sources")
        ds.pop("galaxy_params")
        ds.pop("star_fluxes")

        self.tile_params = {**ds}

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "centered": self.centered[index],
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
