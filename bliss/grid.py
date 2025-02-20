"""Functions for working with `torch.nn.functional.grid_sample`."""

from typing import Callable

import torch
from einops import pack, rearrange, reduce, unpack
from torch import Tensor
from torch.nn.functional import grid_sample


def validate_border_padding(tile_slen: int, ptile_slen: int, bp: int | None = None) -> int:
    # Border Padding
    # Images are first rendered on *padded* tiles (aka ptiles).
    # The padded tile consists of the tile and neighboring tiles
    # The width of the padding is given by ptile_slen.
    # border_padding is the amount of padding we leave in the final image. Useful for
    # avoiding sources getting too close to the edges.
    if bp is not None:
        assert bp == (ptile_slen - tile_slen) / 2
    else:
        bp = (ptile_slen - tile_slen) / 2
    assert float(bp).is_integer(), "amount of border padding must be an integer"
    return int(bp)


def get_mgrid(slen: int, device: torch.device):
    assert slen >= 3
    offset = (slen - 1) / 2
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


def shift_sources_bilinear(
    image_ptiles_flat: Tensor, tile_locs_flat: Tensor, tile_slen: int, slen: int, center=False
) -> Tensor:
    """Shift sources at given padded tiles to given locations.

    The keyword `center` controls whether the sources are already centered at the
    padded tile and should be shifted by `tile_locs_flat` (center=False),
    or if the sources are already shifted by that amount and should be centered (center=True).
    Default is `False`.

    This function can only be used if input image has the same size as padded tiles or is one
    pixel larger. The common use case is that we have odd-sized images (e.g. 53x53) that have a
    light source centered in them but the padded tile is even-sized and one pixel smaller
    (e.g. 52x52). The `grid_sample` function will below will automatically work for both cases of
    input (e.g. 53x53 or 52x52) and in the former case trim the size of the image and correctly
    interpolate.

    An explicit demonstration of this function working correctly can be found in the `experiment`
    notebook: `test-shift-ptiles-fnc.ipynb`.
    """
    npt, _, _, size = image_ptiles_flat.shape
    assert tile_locs_flat.shape[0] == npt
    assert size in {slen, slen + 1}
    bp = validate_border_padding(tile_slen, slen)

    grid = get_mgrid(slen, image_ptiles_flat.device)
    ptile_locs = (tile_locs_flat * tile_slen + bp) / slen
    sgn = 1 if center else -1
    offsets_hw = sgn * (torch.tensor(1.0) - 2 * ptile_locs)
    offsets_xy = swap_locs_columns(offsets_hw)
    grid_inflated = rearrange(grid, "h w xy -> 1 h w xy", xy=2, h=slen)
    offsets_xy_inflated = rearrange(offsets_xy, "npt xy -> npt 1 1 xy", xy=2)
    grid_locs = grid_inflated - offsets_xy_inflated

    sampled_images = grid_sample(image_ptiles_flat, grid_locs, align_corners=True, mode="bilinear")
    assert sampled_images.shape[-1] == sampled_images.shape[-2] == slen
    return sampled_images


def shift_sources(
    images: Tensor,
    locs: Tensor,
    *,
    shift_fnc: Callable,
    tile_slen: int,
    slen: int,
    center: bool = False,
):
    flux = reduce(images, "n 1 h w -> n", "sum")
    mask1 = torch.logical_and(locs[:, 0] > 0, locs[:, 1] > 0)
    mask = torch.logical_and(flux > 0, mask1)

    _images = images[mask]
    _locs_trans = swap_locs_columns(locs[mask])

    sgn = -1 if center else 1
    images_flat = rearrange(_images, "n 1 h w -> n h w")
    offsets = (_locs_trans * tile_slen - tile_slen / 2) * sgn
    shifted_images = shift_fnc(images_flat, offsets)

    final_images = torch.zeros(images.shape[0], 1, slen, slen, device=images.device)
    final_images[mask, 0] = shifted_images
    return final_images
