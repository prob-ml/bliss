"""Functions from producing images from tiled parameters of galaxies or stars."""

import torch
from einops import pack, rearrange, unpack
from galsim import Image, InterpolatedImage
from torch import Tensor
from torch.nn.functional import fold, unfold

from bliss.encoders.autoencoder import CenteredGalaxyDecoder


def render_galaxy_ptiles(
    galaxy_decoder: CenteredGalaxyDecoder,
    locs: Tensor,
    galaxy_params: Tensor,
    galaxy_bools: Tensor,
    tile_slen: int,
    *,
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

    # galsim only runs in CPU
    uncentered_galaxies = _shift_sources_galsim(
        centered_galaxies.to("cpu"),
        locs_flat.to("cpu"),
        galaxy_bools_flat.to("cpu"),
        tile_slen=tile_slen,
        center=False,
    )

    return rearrange(
        uncentered_galaxies, "(b nth ntw) c h w -> b nth ntw c h w", b=b, nth=nth, ntw=ntw
    ).to(locs.device)


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


def _shift_sources_galsim(
    ptiles_flat: Tensor,
    locs: Tensor,
    galaxy_bools: Tensor,
    *,
    tile_slen: int,
    center: bool = False,
    scale: float = 0.2,
):
    """Use interpolation to shift noiseless reconstruction ptiles that contain galaxies."""
    assert locs.ndim == galaxy_bools.ndim == 2
    assert ptiles_flat.shape[1] == 1
    npt = ptiles_flat.shape[0]
    slen = ptiles_flat.shape[-1]
    new_ptiles = torch.zeros_like(ptiles_flat)
    locs_trans = swap_locs_columns(locs)
    sgn = -1 if center else 1

    ptiles_flat_np = ptiles_flat.numpy()
    for ii in range(npt):
        if galaxy_bools[ii].bool().item():
            image = Image(ptiles_flat_np[ii, 0], scale=scale)
            xy = locs_trans[ii]
            offset = (xy * tile_slen - tile_slen / 2) * sgn

            iimg = InterpolatedImage(image, scale=scale)
            fimg = iimg.drawImage(nx=slen, ny=slen, scale=scale, offset=offset, method="no_pixel")
            new_ptiles[ii, 0] = torch.from_numpy(fimg.array)

    return new_ptiles


def reconstruct_image_from_ptiles(image_ptiles: Tensor, tile_slen: int) -> Tensor:
    """Reconstruct an image from padded tiles.

    Args:
        image_ptiles: Tensor of size
            (batch_size x n_tiles_h x n_tiles_w x n_bands x ptile_slen x ptile_slen)

    Returns:
        Reconstructed image of size (batch_size x n_bands x height x width)
    """
    _, nth, ntw, _, ptile_slen, _ = image_ptiles.shape
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
    nth, ntw = get_n_padded_tiles_hw(h, w, tile_slen=tile_slen, ptile_slen=ptile_slen)
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


def crop_ptiles(ptiles: Tensor, locs: Tensor, *, tile_slen: int, bp: int):
    """Make a symmetric croping of images centered around pixel with `loc` with slen 2*bp + 1."""
    assert locs.ndim == 2
    assert not torch.any(locs >= 1.0)
    n, _, h, w = ptiles.shape
    assert h == w

    # find pixel where source is located
    y = locs[:, 0] * tile_slen + bp
    x = locs[:, 1] * tile_slen + bp
    r = (y // 1).long()
    c = (x // 1).long()

    # obtain grids
    xx = torch.arange(h, device=ptiles.device)
    gy, gx = torch.meshgrid(xx, xx, indexing="ij")
    gyy = rearrange(gy, "h w -> 1 1 h w").expand(n, 1, h, w)
    gxx = rearrange(gx, "h w -> 1 1 h w").expand(n, 1, h, w)

    # crop
    cond1 = torch.abs(gyy - rearrange(r, "n -> n 1 1 1")) <= bp
    cond2 = torch.abs(gxx - rearrange(c, "n -> n 1 1 1")) <= bp
    cond = torch.logical_and(cond1, cond2)
    cropped_images = ptiles[cond]

    return rearrange(cropped_images, "(b h w) -> b 1 h w", h=bp * 2 + 1, w=2 * bp + 1)


def get_n_padded_tiles_hw(
    height: int, width: int, *, tile_slen: int, ptile_slen: int
) -> tuple[int, int]:
    nh = ((height - ptile_slen) // tile_slen) + 1
    nw = ((width - ptile_slen) // tile_slen) + 1
    return nh, nw


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


def swap_locs_columns(locs: Tensor) -> Tensor:
    """Swap the columns of locs to invert 'x' and 'y' with einops!"""
    assert locs.ndim == 2 and locs.shape[1] == 2
    x, y = unpack(locs, [[1], [1]], "b *")
    return pack([y, x], "b *")[0]
