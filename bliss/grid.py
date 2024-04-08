"""Functions for working with `torch.nn.functional.grid_sample`."""

import torch
from einops import pack, rearrange, unpack
from torch import Tensor
from torch.nn.functional import grid_sample


def get_mgrid(slen: int, device: torch.device):
    assert slen >= 3 and slen % 2 == 1
    offset = (slen - 1) // 2
    offsets = torch.arange(-offset, offset + 1, 1, device=device)
    x, y = torch.meshgrid(offsets, offsets, indexing="ij")  # same as numpy default indexing.
    mgrid_not_normalized, _ = pack([y, x], "h w *")
    # normalize to (-1, 1) and scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return (mgrid_not_normalized / offset).float() * (slen - 1) / slen


def swap_locs_columns(locs: Tensor) -> Tensor:
    """Swap the columns of locs to invert 'x' and 'y' with einops!"""
    assert locs.ndim == 2 and locs.shape[1] == 2
    x, y = unpack(locs, [[1], [1]], "b *")
    return pack([y, x], "b *")[0]


def center_ptiles(image_ptiles: Tensor, tile_locs_flat: Tensor, tile_slen: int, bp: int) -> Tensor:
    """Center given padded tiles at locations `tile_locs_flat`."""
    npt, _, _, ptile_slen = image_ptiles.shape
    n_ptiles_locs, _ = tile_locs_flat.shape
    assert ptile_slen == image_ptiles.shape[-2]
    assert n_ptiles_locs == npt
    assert bp == (ptile_slen - tile_slen) // 2

    # get new locs to do the shift
    grid = get_mgrid(ptile_slen, image_ptiles.device)
    ptile_locs = (tile_locs_flat * tile_slen + bp) / ptile_slen
    offsets_hw = torch.tensor(1.0) - 2 * ptile_locs
    offsets_xy = swap_locs_columns(offsets_hw)
    grid_inflated = rearrange(grid, "h w xy -> 1 h w xy", xy=2, h=ptile_slen)
    offsets_xy_inflated = rearrange(offsets_xy, "npt xy -> npt 1 1 xy", xy=2)
    grid_loc = grid_inflated - offsets_xy_inflated

    shifted_tiles = grid_sample(image_ptiles, grid_loc, align_corners=True)

    # now that everything is center we can crop easily
    return shifted_tiles[
        ...,
        tile_slen : (ptile_slen - tile_slen),
        tile_slen : (ptile_slen - tile_slen),
    ]
