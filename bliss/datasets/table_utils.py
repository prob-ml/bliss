"""Utilites for turning astrpoy table quantities to Tensors."""

import numpy as np
import torch
from astropy.table import Table


def column_to_tensor(table: Table, colname: str):
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


def table_to_dict(table: Table):
    d = {}
    for p in table.columns:
        d[p] = column_to_tensor(table, p)
    return d


def catsim_row_to_galaxy_params(table: Table, max_n_sources: int):
    names = (
        "fluxnorm_bulge",
        "fluxnorm_disk",
        "fluxnorm_agn",
        "a_b",
        "a_d",
        "b_b",
        "b_d",
        "pa_bulge",
        "i_ab",
        "flux",
    )

    params = torch.zeros((max_n_sources, len(names)))

    for ii, col in enumerate(table):
        for jj, n in enumerate(names):
            params[ii, jj] = column_to_tensor(col, n)

    return params
