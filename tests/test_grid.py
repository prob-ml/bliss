import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn.functional import grid_sample

from bliss.grid import (
    get_mgrid,
    shift_sources_bilinear,
    swap_locs_columns,
)


def _old_get_mgrid(slen: int):
    offset = (slen - 1) / 2

    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset

    return mgrid.float() * (slen - 1) / slen


def _old_centering_tiles(
    image_ptiles: Tensor,
    tile_locs: Tensor,
    ptile_slen: int,
    tile_slen: int,
    bp: int,
):
    swap = torch.tensor([1, 0])
    cached_grid = _old_get_mgrid(ptile_slen)

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


def _shift_sources_in_ptiles(
    locs: Tensor, source: Tensor, ptile_slen: int, tile_slen: int
) -> Tensor:
    """Renders one source at location (per tile) using `grid_sample`."""
    assert source.ndim == 4
    assert source.shape[2] == source.shape[3]
    assert locs.shape[1] == 2 and locs.ndim == 2
    assert locs.device == source.device
    assert ptile_slen >= 2 * tile_slen

    # scale so that they land in the tile within the padded tile
    padding = (ptile_slen - tile_slen) / 2
    ptile_locs = (locs * tile_slen + padding) / ptile_slen
    scaled_locs = (ptile_locs - 0.5) * 2
    locs_swapped = swap_locs_columns(scaled_locs)
    locs_swapped = rearrange(locs_swapped, "np xy -> np 1 1 xy")

    # get grid
    mgrid = get_mgrid(ptile_slen, locs.device)
    local_grid = rearrange(mgrid, "s1 s2 xy -> 1 s1 s2 xy", s1=ptile_slen, s2=ptile_slen, xy=2)
    grid_loc = local_grid - locs_swapped

    return grid_sample(source, grid_loc, align_corners=True)


def test_old_and_new_grid():
    for slen in (3, 4, 5, 6, 7, 10, 11, 26, 51, 52, 53, 43, 44, 101, 102):
        assert torch.all(torch.eq(get_mgrid(slen, torch.device("cpu")), _old_get_mgrid(slen)))


def test_old_and_new_centering():
    ps = 53
    ts = 5
    ptiles = torch.randn((10, 1, ps, ps)) * 10 + 100
    tile_locs = torch.rand((1, 10, 1, 2))

    centered_ptiles2 = _shift_sources_in_ptiles(tile_locs[0, :, 0, :], ptiles, ps, ts)

    centered_ptiles3 = shift_sources_bilinear(ptiles, tile_locs[0, :, 0, :], ts, ps, center=True)
    centered_ptiles4 = shift_sources_bilinear(ptiles, tile_locs[0, :, 0, :], ts, ps, center=False)

    assert centered_ptiles2.ndim == centered_ptiles3.ndim == 4
    assert centered_ptiles3.shape[-1] == ps
    assert centered_ptiles4.shape[-1] == ps
    assert torch.all(torch.eq(centered_ptiles2, centered_ptiles4))
