import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from bliss.render_tiles import reconstruct_image_from_ptiles


def test_reconstruct_image_from_ptiles():
    ptiles = torch.randn((32, 10, 10, 1, 52, 52)) * 10 + 100

    images = reconstruct_image_from_ptiles(ptiles, tile_slen=4)

    assert images.shape == (32, 1, 88, 88)


def test_trim_source():
    source = torch.randn((1, 53, 53))

    new_source1 = old_trim_source(source, 52)
    new_source = _trim_source(source, 52)
    new_source2 = _fit_source_to_ptile(source, 52)
    new_source3 = _size_galaxy(source[None, :, :, :], 52)

    assert new_source.shape == new_source1.shape
    assert new_source.shape == new_source2.shape
    assert new_source.shape == new_source3[0].shape

    x = 53
    y = 52
    assert x + ((x % 2) == 0) * 1 == 53
    assert y + ((y % 2) == 0) * 1 == 53


def old_trim_source(source, ptile_slen: int):
    """Crop the source to length ptile_slen x ptile_slen, centered at the middle."""
    assert len(source.shape) == 3

    # if self.ptile_slen is even, we still make source dimension odd.
    # otherwise, the source won't have a peak in the center pixel.
    local_slen = ptile_slen + ((ptile_slen % 2) == 0) * 1

    source_slen = source.shape[2]
    source_center = (source_slen - 1) / 2

    assert source_slen >= local_slen

    r = np.floor(local_slen / 2)
    l_indx = int(source_center - r)
    u_indx = int(source_center + r + 1)

    return source[:, l_indx:u_indx, l_indx:u_indx]


def _size_galaxy(galaxy: Tensor, ptile_slen: int):
    n, c, h, w = galaxy.shape
    assert h == w
    assert (h % 2) == 1, "dimension of galaxy image should be odd"
    galaxy = rearrange(galaxy, "n c h w -> (n c) h w")
    sized_galaxy = _fit_source_to_ptile(galaxy, ptile_slen)
    return rearrange(sized_galaxy, "(n c) h w -> n c h w", n=n, c=c)


def _fit_source_to_ptile(source: Tensor, ptile_slen: int) -> Tensor:
    if ptile_slen >= source.shape[-1]:
        fitted_source = _expand_source(source, ptile_slen)
    else:
        fitted_source = _trim_source(source, ptile_slen)
    return fitted_source


def _expand_source(source: Tensor, ptile_slen: int) -> Tensor:
    """Pad the source with zeros so that it is size ptile_slen."""
    assert source.ndim == 3
    slen = ptile_slen if ptile_slen % 2 == 1 else ptile_slen + 1
    source_slen = source.shape[2]

    assert source_slen <= slen, "Should be using trim source."

    source_expanded = torch.zeros(source.shape[0], slen, slen, device=source.device)
    offset = int((slen - source_slen) / 2)

    source_expanded[:, offset : (offset + source_slen), offset : (offset + source_slen)] = source

    return source_expanded


def _trim_source(source: Tensor, ptile_slen: int) -> Tensor:
    """Crop the source to dimensions `ptile_slen`, centered at the middle."""
    assert source.ndim == 3

    # if self.ptile_slen is even, we still make source dimension odd.
    # otherwise, the source won't have a peak in the center pixel.
    local_slen = ptile_slen if ptile_slen % 2 == 1 else ptile_slen + 1

    source_slen = source.shape[2]
    source_center = (source_slen - 1) / 2

    assert source_slen >= local_slen

    r = np.floor(local_slen / 2)
    l_indx = int(source_center - r)
    u_indx = int(source_center + r + 1)

    return source[:, l_indx:u_indx, l_indx:u_indx]
