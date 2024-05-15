"""Functions from producing images from tiled parameters of galaxies or stars."""

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.functional import fold, unfold

from bliss.encoders.autoencoder import CenteredGalaxyDecoder
from bliss.grid import shift_sources_in_ptiles


def render_galaxy_ptiles(
    galaxy_decoder: CenteredGalaxyDecoder,
    locs: Tensor,
    galaxy_params: Tensor,
    galaxy_bools: Tensor,
    ptile_slen: int,
    tile_slen: int,
    n_bands: int = 1,
) -> Tensor:
    """Render padded tiles of galaxies from tiled tensors."""
    assert galaxy_bools.le(1).all(), "At most one source can be rendered per tile."
    b, nth, ntw, _ = locs.shape

    locs_flat = rearrange(locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
    galaxy_bools_flat = rearrange(galaxy_bools, "b nth ntw 1 -> (b nth ntw) 1")
    galaxy_params_flat = rearrange(galaxy_params, "b nth ntw d -> (b nth ntw) d")

    # NOTE: size of tiles here is galaxy_decoder size, not `ptile_slen`!.
    centered_galaxies = _render_centered_galaxies_ptiles(
        galaxy_decoder, galaxy_params_flat, galaxy_bools_flat, n_bands
    )
    assert centered_galaxies.shape[-1] == galaxy_decoder.slen
    assert galaxy_decoder.slen % 2 == 1  # so centered in central pixel

    # render galaxies in correct location within padded tile and trim to be size `ptile_slen`
    uncentered_galaxies = shift_sources_in_ptiles(
        centered_galaxies, locs_flat, tile_slen, ptile_slen, center=False
    )
    assert uncentered_galaxies.shape[-1] == ptile_slen

    return rearrange(
        uncentered_galaxies, "(b nth ntw) c h w -> b nth ntw c h w", b=b, nth=nth, ntw=ntw
    )


def _render_centered_galaxies_ptiles(
    galaxy_decoder: CenteredGalaxyDecoder,
    galaxy_params: Tensor,
    galaxy_bools: Tensor,
    n_bands: int = 1,
) -> Tensor:
    assert galaxy_params.ndim == galaxy_bools.ndim == 2
    n_ptiles, _ = galaxy_params.shape
    is_gal = galaxy_bools.flatten().bool()

    # forward only galaxies that are on!
    gal_on = galaxy_decoder(galaxy_params[is_gal])
    size = gal_on.shape[-1]

    # allocate memory
    gal = torch.zeros(n_ptiles, n_bands, size, size, device=galaxy_params.device)

    # set galaxies
    gal[is_gal] = gal_on

    # be extra careful
    return gal * rearrange(is_gal, "npt -> npt 1 1 1")


def reconstruct_image_from_ptiles(image_ptiles: Tensor, tile_slen: int) -> Tensor:
    """Reconstruct an image from padded tiles.

    Args:
        image_ptiles: Tensor of size
            (batch_size x n_tiles_h x n_tiles_w x n_bands x ptile_slen x ptile_slen)

    Returns:
        Reconstructed image of size (batch_size x n_bands x height x width)
    """
    _, nth, ntw, _, ptile_slen, _ = image_ptiles.shape  # noqa: WPS236
    image_ptiles_prefold = rearrange(image_ptiles, "b nth ntw c h w -> b (c h w) (nth ntw)")
    kernel_size = (ptile_slen, ptile_slen)
    stride = (tile_slen, tile_slen)
    nthw = (nth, ntw)

    output_size_list = []
    for i in (0, 1):
        output_size_list.append(kernel_size[i] + (nthw[i] - 1) * stride[i])
    output_size = tuple(output_size_list)

    # In default settings, no borders are cropped from output image.
    folded_image = fold(image_ptiles_prefold, output_size, kernel_size, stride=stride)
    assert folded_image.shape[-2] == tile_slen * nth + (ptile_slen - tile_slen)
    return folded_image


def get_images_in_tiles(images: Tensor, tile_slen: int, ptile_slen: int) -> Tensor:
    """Divides a batch of full images into padded tiles.

    This is similar to nn.conv2d, with a sliding window=ptile_slen and stride=tile_slen.

    Arguments:
        images: Tensor of images with size (batchsize x n_bands x slen x slen)
        tile_slen: Side length of tile
        ptile_slen: Side length of padded tile

    Returns:
        A batchsize x nth x ntw x n_bands x tile_height x tile_width image
    """
    assert images.ndim == 4
    _, c, h, w = images.shape
    nth, ntw = get_n_padded_tiles_hw(h, w, ptile_slen, tile_slen)
    tiles = unfold(images, kernel_size=ptile_slen, stride=tile_slen)
    return rearrange(
        tiles,
        "b (c pth ptw) (nth ntw) -> b nth ntw c pth ptw",
        nth=nth,
        ntw=ntw,
        c=c,
        pth=ptile_slen,
        ptw=ptile_slen,
    )


def get_n_padded_tiles_hw(
    height: int, width: int, ptile_slen: int, tile_slen: int
) -> tuple[int, int]:
    nh = ((height - ptile_slen) // tile_slen) + 1
    nw = ((width - ptile_slen) // tile_slen) + 1
    return nh, nw
