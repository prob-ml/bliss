import numpy as np
import torch
from torch import Tensor


def convert_mag_to_flux(mag: Tensor) -> Tensor:
    raise NotImplementedError


def convert_flux_to_mag(flux: Tensor) -> Tensor:
    raise NotImplementedError


def column_to_tensor(table, colname):
    dtypes = {
        np.dtype(">i2"): int,
        np.dtype(">i4"): int,
        np.dtype(">i8"): int,
        np.dtype("bool"): bool,
        np.dtype(">f4"): np.float32,
        np.dtype(">f8"): np.float32,
        np.dtype("float32"): np.float32,
        np.dtype("float64"): np.dtype("float64"),
    }
    x = np.array(table[colname])
    dtype = dtypes[x.dtype]
    x = x.astype(dtype)
    return torch.from_numpy(x)
