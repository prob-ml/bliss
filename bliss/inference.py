import math
from typing import Dict, List, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from bliss.datasets.sdss_blended_galaxies import cpu
from bliss.encoder import Encoder
from bliss.models.decoder import ImageDecoder
from bliss.models.location_encoder import get_full_params_from_tiles, get_params_in_batches


def reconstruct_scene_at_coordinates(
    encoder: Encoder,
    decoder: ImageDecoder,
    img: Tensor,
    h: int,
    w: int,
    scene_length: int,
    slen: int = 80,
    bp: int = 24,
    device=None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Reconstruct all objects contained within a scene, padding as needed.

    This function will run the encoder and decoder on a padded image containing the image specified
    via h, w, and scene_length, to ensure that all objects can be detected and reconstructed.

    Args:
        encoder:
            Trained Encoder module.
        decoder:
            Trained ImageDecoder module.
        img:
            A NxCxHxW tensor of N HxW images (with C bands).
        h:
            Starting height coordinate (top-left)
        w:
            Starting width coordinate (top-left)
        scene_length:
            Size of (square) image to reconstruct.
        slen:
            The side-lengths of smaller chunks to create. Defaults to 80.
        bp:
            Border padding needed by encoder. Defaults to 24.
        device:
            Device used for rendering each chunk (i.e. a torch.device). Note
            that chunks are moved onto and off the device to allow for rendering
            larger images.

    Returns:
        A tuple of two items:
        -  recon_at_coords: A NxCxHxW Tensor of the reconstructed images
            at the coordinates specified hy h, w, and scene_length.
        -  map_scene: The maximum-a-posteriori catalog estimated from the image
            (and surrounding border padding). Note that this may contain objects detected
            outside the bounds of the image given, but their coordinates will be relative to (h, w).
            In other words, the locations of these out-of-bounds objects will either be negative if
            they are to the left/above (h, w) or greater than scene_length.

    """
    # Adjust bp so that we get an even number of chunks in scene
    extra = (scene_length - bp * 2) % slen
    while extra < (2 * bp):
        extra += slen
    adj_scene_length = scene_length + extra
    if h + adj_scene_length - bp > img.shape[2]:
        h_padded = img.shape[2] - adj_scene_length
    else:
        h_padded = h - bp
    if w + adj_scene_length - bp > img.shape[3]:
        w_padded = img.shape[3] - adj_scene_length
    else:
        w_padded = w - bp

    # First get the mininum coordinates to ensure everything is detected
    scene = img[
        :, :, h_padded : (h_padded + adj_scene_length), w_padded : (w_padded + adj_scene_length)
    ]
    assert scene.shape[2] == scene.shape[3] == adj_scene_length

    recon, map_scene = reconstruct_scene(encoder, decoder, scene, slen, bp, device=device)

    # Get reconstruction at coordinates
    recon_at_coords = recon[
        :,
        :,
        (h - h_padded) : (h - h_padded + scene_length),
        (w - w_padded) : (w - w_padded + scene_length),
    ]

    # Adjust locations based on padding
    h_adj = h_padded - (h - bp)
    w_adj = w_padded - (w - bp)
    plocs = map_scene["plocs"]
    plocs[..., 0] += h_adj
    plocs[..., 1] += w_adj
    map_scene["plocs"] = plocs

    return recon_at_coords, map_scene


def reconstruct_scene(
    encoder: Encoder,
    decoder: ImageDecoder,
    scene: Tensor,
    slen: int = 80,
    bp: int = 24,
    device=None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Perform reconstruction and MAP estimation chunk-by-chunk on given scene.

    Currently this function only works on square images and assumes they have been appropriately
    padded (as in `reconstruct_scene_at_coordinates`) so that it can be divided evenly into chunks.

    Args:
        encoder:
            Trained Encoder module.
        decoder:
            Trained ImageDecoder module.
        scene:
            A NxCxLxL tensor of N LxL images (with C bands).
        slen:
            The side-lengths of smaller chunks to create. Defaults to 80.
        bp:
            Border padding needed by encoder. Defaults to 24.
        device:
            Device used for rendering each chunk (i.e. a torch.device). Note
            that chunks are moved onto and off the device to allow for rendering
            larger images.

    Returns:
        A tuple of two items:
            - The reconstruction of the given scene `scene_recon`.
            - The map estimate of parameters of each detected source on the scene `map_recon`.
    """
    if device is None:
        device = torch.device("cpu")
    chunks = split_scene_into_chunks(scene, slen, bp)
    reconstructions = []
    bgs = []
    full_maps = []
    for chunk in tqdm(chunks):
        recon, bg, full_map = reconstruct_img(encoder, decoder, chunk.unsqueeze(0).to(device))
        reconstructions.append(recon.cpu())
        bgs.append(bg.cpu())
        full_maps.append(cpu(full_map))
    reconstructions = torch.cat(reconstructions, dim=0)
    bgs = torch.cat(bgs, dim=0)
    scene_recon = combine_chunks_into_scene(reconstructions, bgs, slen)
    map_recon = combine_full_maps(full_maps, slen)

    return scene_recon, map_recon


def split_scene_into_chunks(scene: Tensor, slen: int, bp: int) -> Tensor:
    """Split scenes into square chunks of side length `slen+bp*2` using `F.unfold`."""
    kernel_size = slen + bp * 2
    chunks = F.unfold(scene, kernel_size=kernel_size, stride=slen)
    return rearrange(
        chunks,
        "b (c h w) n -> (b n) c h w",
        c=scene.shape[1],
        h=kernel_size,
        w=kernel_size,
    )


def reconstruct_img(
    encoder: Encoder, decoder: ImageDecoder, img: Tensor
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    img_ptiles = encoder.get_images_in_ptiles(img)

    with torch.no_grad():
        tile_map = encoder.max_a_post(img_ptiles)
        tile_map = get_params_in_batches(tile_map, img.shape[0])
        recon_image, _ = decoder.render_images(
            tile_map["n_sources"],
            tile_map["locs"],
            tile_map["galaxy_bools"],
            tile_map["galaxy_params"],
            tile_map["fluxes"],
            add_noise=False,
        )
        background = decoder.get_background(recon_image.shape[-1]).unsqueeze(0)
        tile_map["galaxy_fluxes"] = decoder.get_galaxy_fluxes(
            tile_map["galaxy_bools"], tile_map["galaxy_params"]
        )
        full_map = get_full_params_from_tiles(tile_map, decoder.tile_slen)
    return recon_image, background, full_map


def combine_chunks_into_scene(recon_chunks: Tensor, bgs: Tensor, slen: int):
    kernel_size = recon_chunks.shape[-1]
    bp = kernel_size - slen
    rr = rearrange(recon_chunks - bgs, "(b n) c h w -> b (c h w) n", b=1, c=1)
    n_tiles_h = int(math.sqrt(recon_chunks.shape[0]))
    output_size = kernel_size + (n_tiles_h - 1) * slen
    rrr = F.fold(rr, output_size=output_size, kernel_size=kernel_size, stride=slen)
    # We now need to add back in the background; we zero out the right and bottom borders
    # to avoid double-counting
    bgs = rearrange(bgs, "(b nh nw) c h w -> b nh nw c h w", b=1, nh=n_tiles_h)
    bgs[:, :-1, :, :, -bp:, :] = 0.0
    bgs[:, :, :-1, :, :, -bp:] = 0.0
    bgr = rearrange(bgs, "b nh nw c h w -> b (c h w) (nh nw)", b=1, c=1)
    bgrrr = F.fold(bgr, output_size=output_size, kernel_size=kernel_size, stride=slen)
    return rrr + bgrrr


def combine_full_maps(full_maps: List[Dict[str, Tensor]], chunk_slen: int) -> Dict[str, Tensor]:
    """Combine full maps of chunks into one dictionary for whole scene.

    Args:
        full_maps: A list of dictionaries where each element is the output
            of get_full_params_from_tiles() on a particular chunk.
        chunk_slen:
            The side-length of each chunk (assumed to be square chunks).


    Returns:
        A dictionary of tensors of size n_samples x sum(max_sources) x ..., where
        sum(max_sources) is the sum of the maximum sources found on each chunk.
        The members of the dictionary are equivalent to those in get_full_params_from_tiles().

        Note that the output of this function is not sorted; you must rely on galaxy_bools and
        star_bools to determine if a particular source is actually on or not.
    """

    params = {}
    for k in full_maps[0].keys():
        tensors = [full_map[k] for full_map in full_maps]
        if k != "n_sources":
            params[k] = torch.cat(tensors, dim=1)
        else:
            n_sources = torch.stack(tensors, dim=0)

    n_chunks_h = int(math.sqrt(len(full_maps)))
    n_chunks_w = n_chunks_h
    assert n_chunks_h * n_chunks_w == len(full_maps)

    offsets_h = torch.tensor(range(n_chunks_h), device=params["locs"].device)
    offsets_w = torch.tensor(range(n_chunks_w), device=params["locs"].device)
    offsets = torch.cartesian_prod(offsets_h, offsets_w)
    pbias = []

    for i, offset in enumerate(offsets):
        max_sources = full_maps[i]["locs"].shape[1]
        pbias_i = repeat(offset * chunk_slen, "xy -> 1 max_sources xy", max_sources=max_sources)
        n_sources_i = n_sources[i].unsqueeze(-1)

        mask_i = torch.tensor(range(max_sources))
        mask_i = repeat(mask_i, "max_sources -> n max_sources", n=n_sources_i.shape[0])
        mask_i = mask_i < n_sources_i
        pbias_i *= mask_i.unsqueeze(-1)

        pbias.append(pbias_i)

    pbias = torch.cat(pbias, dim=1)
    params["plocs"] = params["plocs"] + pbias
    params["locs"][..., 0] = params["locs"][..., 0] / (chunk_slen * n_chunks_h)
    params["locs"][..., 1] = params["locs"][..., 1] / (chunk_slen * n_chunks_w)

    return params
