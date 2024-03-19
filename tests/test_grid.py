import numpy as np
import torch

from bliss.grid import get_mgrid


def old_get_mgrid(slen: int):
    offset = (slen - 1) / 2

    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset

    return mgrid.float() * (slen - 1) / slen


def test_old_and_new_grid():
    for slen in (3, 5, 7, 11, 51, 53, 43, 101):
        assert torch.all(torch.eq(get_mgrid(slen, torch.device("cpu")), old_get_mgrid(slen)))
