"""Functions from producing images from tiled parameters of galaxies or stars."""

import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.nn.functional import fold, unfold
from tqdm import tqdm

from bliss.datasets.lsst import PIXEL_SCALE
from bliss.encoders.autoencoder import CenteredGalaxyDecoder
from bliss.grid import shift_sources_in_ptiles
from bliss.reporting import get_single_galaxy_ellipticities


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


def get_galaxy_fluxes(
    galaxy_decoder: CenteredGalaxyDecoder, galaxy_bools: Tensor, galaxy_params_in: Tensor
) -> Tensor:

    # obtain galaxies from decoder and latents
    galaxy_bools_flat = rearrange(galaxy_bools, "b nth ntw 1 -> (b nth ntw 1)")
    galaxy_params = rearrange(galaxy_params_in, "b nth ntw d -> (b nth ntw) d")
    is_gal = torch.ge(galaxy_bools_flat, 0.5)
    galaxy_shapes = galaxy_decoder(galaxy_params[is_gal])

    # get fluxes and reshape
    galaxy_fluxes = reduce(galaxy_shapes, "n 1 h w -> n", "sum")
    assert torch.all(galaxy_fluxes >= 0)
    galaxy_fluxes_all = torch.zeros_like(galaxy_bools_flat, dtype=galaxy_fluxes.dtype)
    galaxy_fluxes_all[is_gal] = galaxy_fluxes
    galaxy_fluxes_all = rearrange(galaxy_fluxes_all, "(b nth ntw) -> b nth ntw 1")
    return galaxy_fluxes_all * galaxy_bools


def get_galaxy_ellips(
    galaxy_decoder: CenteredGalaxyDecoder,
    galaxy_bools: Tensor,
    galaxy_params_in: Tensor,
    psf: Tensor,
) -> Tensor:
    assert galaxy_bools.ndim == 4 and galaxy_params_in.ndim == 4
    assert psf.shape == (1, galaxy_decoder.slen, galaxy_decoder.slen)
    b, nth, ntw, _ = galaxy_bools.shape
    b_flat = b * nth * ntw

    is_gal = rearrange(galaxy_bools, "b nth ntw 1 -> (b nth ntw 1)").bool()
    galaxy_params = rearrange(galaxy_params_in, "b nth ntw d -> (b nth ntw) d")
    galaxy_shapes = galaxy_decoder.forward(galaxy_params[is_gal]).detach().cpu()
    single_galaxies = rearrange(galaxy_shapes, "n 1 h w -> n h w")
    assert galaxy_shapes.shape[-1] == galaxy_shapes.shape[-2] == galaxy_decoder.slen

    ellips_gal = get_single_galaxy_ellipticities(single_galaxies, psf[0], PIXEL_SCALE)

    ellips_all = torch.zeros(b_flat, 2, dtype=ellips_gal.dtype, device=ellips_gal.device)
    ellips_all[is_gal] = ellips_gal
    ellips = rearrange(ellips_all, "(b nth ntw s) g -> b nth ntw g", b=b, nth=nth, ntw=ntw, g=2)
    return ellips * galaxy_bools


def get_galaxy_snr(self, decoder: CenteredGalaxyDecoder, ptile_slen: int, bg: float) -> None:

    b, nth, ntw, _ = self["galaxy_params"].shape
    galaxy_params = rearrange(self["galaxy_params"], "b nth ntw d -> (b nth ntw) d")
    galaxy_bools = rearrange(self["galaxy_bools"], "b nth ntw 1 -> (b nth ntw) 1")

    n_total = b * nth * ntw
    snr = torch.zeros((n_total, 1))
    n_parts = 1000  # not all fits in GPU
    for ii in tqdm(range(0, n_total, n_parts), desc="computing SNR", total=n_total // n_parts):
        jj = ii + n_parts
        galaxy_params_ii = galaxy_params[ii:jj]
        galaxy_bools_ii = galaxy_bools[ii:jj]

        # galaxies first to get correct sizes
        single_galaxies_ii = _render_centered_galaxies_ptiles(
            decoder, galaxy_params_ii, galaxy_bools_ii, ptile_slen
        )
        single_galaxies_ii = rearrange(single_galaxies_ii, "np s 1 h w -> np s h w")
        single_galaxies_ii = single_galaxies_ii.cpu()
        _, _, h, w = single_galaxies_ii.shape
        assert h == w
        bg_ii = torch.full_like(single_galaxies_ii, bg)

        gal_snr = (single_galaxies_ii**2 / (single_galaxies_ii + bg_ii)).sum(axis=(-1, -2)).sqrt()

        gal_snr = rearrange(gal_snr, "n s -> n s 1")

        for kk in range(ii, jj):
            snr[kk] = gal_snr[kk] * galaxy_bools_ii[kk].cpu()

    return rearrange(snr, "(b nth ntw) s 1 -> b nth ntw s 1", b=b, nth=nth, ntw=ntw)
