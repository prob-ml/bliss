import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn.functional import grid_sample

from bliss.grid import center_ptiles, get_mgrid


def old_get_mgrid(slen: int):
    offset = (slen - 1) / 2

    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset

    return mgrid.float() * (slen - 1) / slen


def old_centering_tiles(
    image_ptiles: Tensor,
    tile_locs: Tensor,
    ptile_slen: int,
    tile_slen: int,
    bp: int,
):
    swap = torch.tensor([1, 0])
    cached_grid = old_get_mgrid(ptile_slen)

    n_ptiles, _, _, ptile_slen_img = image_ptiles.shape
    n_samples, n_ptiles_locs, max_sources, _ = tile_locs.shape
    assert max_sources == 1
    assert ptile_slen_img == ptile_slen
    assert n_ptiles_locs == n_ptiles

    # get new locs to do the shift
    ptile_locs = (tile_locs * tile_slen + bp) / ptile_slen
    offsets_hw = torch.tensor(1.0) - 2 * ptile_locs
    offsets_xy = offsets_hw.index_select(dim=-1, index=swap)
    grid_loc = cached_grid.view(1, ptile_slen, ptile_slen, 2) - offsets_xy.view(-1, 1, 1, 2)

    # Expand image_ptiles to match number of samples
    image_ptiles = image_ptiles.unsqueeze(0).expand(n_samples, -1, -1, -1, -1)
    image_ptiles = image_ptiles.reshape(
        n_samples * n_ptiles,
        -1,
        ptile_slen,
        ptile_slen,
    )
    shifted_tiles = grid_sample(image_ptiles, grid_loc, align_corners=True)

    # now that everything is center we can crop easily
    shifted_tiles = shifted_tiles[
        :,
        :,
        tile_slen : (ptile_slen - tile_slen),
        tile_slen : (ptile_slen - tile_slen),
    ]
    return rearrange(shifted_tiles, "(ns np) c h w -> ns np c h w", ns=n_samples, np=n_ptiles)


def test_old_and_new_grid():
    for slen in (3, 4, 5, 6, 7, 10, 11, 26, 51, 52, 53, 43, 44, 101, 102):
        assert torch.all(torch.eq(get_mgrid(slen, torch.device("cpu")), old_get_mgrid(slen)))


def test_old_and_new_centering():
    ptiles = torch.randn((10, 1, 52, 52)) * 10 + 10
    tile_locs = torch.rand((1, 10, 1, 2))

    centered_ptiles1 = old_centering_tiles(ptiles, tile_locs, 52, 4, 24)
    centered_ptiles2 = center_ptiles(ptiles, tile_locs[0, :, 0, :], 4, 24)

    assert torch.all(torch.eq(centered_ptiles1[0], centered_ptiles2))
