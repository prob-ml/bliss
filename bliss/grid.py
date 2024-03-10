import numpy as np
import torch


def get_mgrid(slen: int):
    offset = (slen - 1) / 2
    # Currently type-checking with mypy doesn't work with np.mgrid
    # See https://github.com/python/mypy/issues/11185.
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]  # type: ignore
    mgrid = torch.from_numpy(np.dstack((y, x))) / offset
    # mgrid is between -1 and 1
    # then scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return mgrid.float() * (slen - 1) / slen
