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
        np.dtype(">f8"): np.float32,  # convert 64-bit precision float to 32-bit precision
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
    n_rows = len(table)
    assert n_rows <= max_n_sources

    names = (
        "fluxnorm_bulge",
        "fluxnorm_disk",
        "fluxnorm_agn",
        "a_b",
        "a_d",
        "b_b",
        "b_d",
        "pa_bulge",
        "pa_disk",
        "i_ab",
        "flux",
    )

    params = torch.zeros((max_n_sources, len(names)), dtype=torch.float32)

    for jj, n in enumerate(names):
        dtype = table[n].dtype
        # everything should be a float32 or float64
        assert dtype == np.dtype(">f8") or dtype == np.float32  # noqa:WPS514
        params[:n_rows, jj] = column_to_tensor(table, n)

    return params
