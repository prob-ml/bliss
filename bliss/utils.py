import numpy as np
import torch
from torch import Tensor

from bliss.catalog import FullCatalog


def convert_mag_to_flux(mag: Tensor, nelec_per_nmgy=987.31) -> Tensor:
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 10 ** ((22.5 - mag) / 2.5) * nelec_per_nmgy


def convert_flux_to_mag(flux: Tensor, nelec_per_nmgy=987.31) -> Tensor:
    # default corresponds to average value of columns for run 94, camcol 1, field 12
    return 22.5 - 2.5 * torch.log10(flux / nelec_per_nmgy)


def get_mags_from_fluxes(cat: FullCatalog) -> Tensor:
    fluxes = torch.where(
        cat["star_bools"].squeeze().to(bool),  # type: ignore
        cat["star_fluxes"].squeeze(),
        cat["galaxy_params"][..., 0].squeeze(),
    )[..., None]
    return convert_flux_to_mag(fluxes)


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
