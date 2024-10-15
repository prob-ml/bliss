"""Methods to save datasets using h5py."""

# useful resource: https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def save_dataset_npz(ds: dict[str, Tensor], fpath: str | Path) -> None:
    assert not Path(fpath).exists(), "overwriting existing ds"
    assert Path(fpath).suffix == ".npz"
    ds_np = {k: v.numpy() for k, v in ds.items()}
    np.savez(fpath, **ds_np)


def load_dataset_npz(fpath: str | Path) -> dict[str, Tensor]:
    assert Path(fpath).exists(), "file path does not exists"
    ds = {}
    npzfile = np.load(fpath)
    for k in npzfile.files:
        ds[k] = torch.from_numpy(npzfile[k])
    return ds
