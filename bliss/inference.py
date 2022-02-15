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
from bliss.models.location_encoder import get_full_params_from_tiles


def reconstruct_scene_at_coordinates(
    encoder: Encoder,
    decoder: ImageDecoder,
    img: Tensor,
    h_range: Tuple[int, int],
    w_range: Tuple[int, int],
    slen: int = 80,
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
        h_range:
            Range of height coordinates.
        w_range:
            Range of width coordinates.
        scene_length:
            Size of (square) image to reconstruct.
        slen:
            The side-lengths of smaller chunks to create. Defaults to 80.
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
    bp = encoder.border_padding
    h_range_pad = (h_range[0] - bp, h_range[1] + bp)
    w_range_pad = (w_range[0] - bp, w_range[1] + bp)
    # extra = (scene_length - bp * 2) % slen
    # while extra < (2 * bp):
    #     extra += slen
    # adj_scene_length = scene_length + extra
    # if h + adj_scene_length - bp > img.shape[2]:
    #     h_padded = img.shape[2] - adj_scene_length
    # else:
    #     h_padded = h - bp
    # if w + adj_scene_length - bp > img.shape[3]:
    #     w_padded = img.shape[3] - adj_scene_length
    # else:
    #     w_padded = w - bp

    # First get the mininum coordinates to ensure everything is detected
    scene = img[:, :, h_range_pad[0] : h_range_pad[1], w_range_pad[0] : w_range_pad[1]]
    assert scene.shape[2] == h_range_pad[1] - h_range_pad[0]
    assert scene.shape[3] == w_range_pad[1] - w_range_pad[0]

    recon, map_scene = reconstruct_scene(encoder, decoder, scene, slen=slen, device=device)

    # Get reconstruction at coordinates
    recon_at_coords = recon[
        :,
        :,
        bp:-bp,
        bp:-bp,
    ]

    # Adjust locations based on padding
    # h_adj = h_padded - (h - bp)
    # w_adj = w_padded - (w - bp)
    plocs = map_scene["plocs"]
    plocs[..., 0] -= bp
    plocs[..., 1] -= bp
    map_scene["plocs"] = plocs

    return recon_at_coords, map_scene


def reconstruct_scene(
    encoder: Encoder,
    decoder: ImageDecoder,
    scene: Tensor,
    slen: int = 80,
    device=None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if device is None:
        device = torch.device("cpu")
    chunks, offsets = split_scene_into_chunks(scene, slen, encoder.border_padding)
    reconstructions = []
    bgs = []
    full_maps = []
    for chunk in tqdm(chunks):
        if chunk is not None:
            recon, bg, full_map = reconstruct_img(encoder, decoder, chunk.unsqueeze(0).to(device))
        else:
            recon, bg, full_map = None, None, None
        reconstructions.append(recon.cpu())
        bgs.append(bg.cpu())
        full_maps.append(cpu(full_map))
    # reconstructions = torch.cat(reconstructions, dim=0)
    # bgs = torch.cat(bgs, dim=0)
    scene_recon = combine_chunks_into_scene(reconstructions, bgs, offsets, slen)
    map_recon = combine_full_maps(full_maps, offsets, slen)

    return scene_recon, map_recon


class ChunkedScene:
    def __init__(self, scene: Tensor, slen: int, bp: int):
        """Split scenes into square chunks of side length `slen+bp*2` using `F.unfold`."""
        self.slen = slen
        self.bp = bp
        kernel_size = slen + bp * 2
        self.chunk_dict = {}
        self.size_dict = {}

        n_chunks_h = (scene.shape[2] - (bp * 2)) // slen
        n_chunks_w = (scene.shape[3] - (bp * 2)) // slen
        self.chunk_dict["main"] = self._chunk_image(scene, kernel_size, slen)
        self.size_dict["main"] = (n_chunks_h * slen + bp * 2, n_chunks_w * slen + bp * 2)

        offsets_h = torch.tensor(range(n_chunks_h))
        offsets_w = torch.tensor(range(n_chunks_w))
        offsets = torch.cartesian_prod(offsets_h, offsets_w)
        self.biases["main"] = offsets * self.slen

        # Get leftover chunks
        bottom_border_start = slen * n_chunks_h - bp
        bottom_chunk_height = scene.shape[2] - bottom_border_start
        assert bottom_chunk_height > bp * 2

        right_border_start = slen * n_chunks_w - bp
        right_chunk_width = scene.shape[3] - right_border_start
        assert right_chunk_width > bp * 2

        bottom_border = scene[:, :, bottom_border_start:, :right_border_start]
        self.chunk_dict["bottom"] = self._chunk_image(
            bottom_border, (bottom_chunk_height, kernel_size), slen
        )
        self.size_dict["bottom"] = bottom_border.shape[2:]
        self.biases["bottom"] = (
            torch.cartesian_prod(offsets_h, torch.tensor([n_chunks_w])) * self.slen
        )

        right_border = scene[:, :, :bottom_border_start, right_border_start:]
        self.chunk_dict["right"] = self._chunk_image(
            right_border, (kernel_size, right_chunk_width), slen
        )
        self.size_dict["right"] = right_border.shape[2:]
        self.biases["right"] = (
            torch.cartesian_prod(torch.tensor([n_chunks_h]), offsets_w) * self.slen
        )

        bottom_right_border = scene[:, :, bottom_border_start:, right_border_start:]
        self.chunk_dict["bottom_right"] = [bottom_right_border]
        self.size_dict["bottom_right"] = bottom_right_border.shape[2:]
        self.biases["bottom_right"] = (
            torch.cartesian_prod(torch.tensor([n_chunks_h]), torch.tensor([n_chunks_w])) * self.slen
        )

    def _chunk_image(self, image, kernel_size, stride):
        chunks = F.unfold(image, kernel_size=kernel_size, stride=stride)
        chunks_rearranged = rearrange(
            chunks,
            "b (c h w) n -> (b n) c h w",
            c=image.shape[1],
            h=kernel_size,
            w=kernel_size,
        )
        return list(chunks_rearranged)

    def reconstruct(self, encoder, decoder, device):
        chunk_est_dict = {}
        for chunk_type, chunks in self.chunk_dict.items():
            chunk_est_dict[chunk_type] = self._reconstruct_chunks(chunks, encoder, decoder, device)
        scene_recon = self._combine_into_scene(chunk_est_dict)
        map_recon = self._combine_full_maps(chunk_est_dict)
        return scene_recon, map_recon

    def _reconstruct_chunks(self, chunks, encoder, decoder, device):
        reconstructions = []
        bgs = []
        full_maps = []
        for chunk in tqdm(chunks):
            recon, bg, full_map = reconstruct_img(encoder, decoder, chunk.unsqueeze(0).to(device))
            reconstructions.append(recon.cpu())
            bgs.append(bg.cpu())
            full_maps.append(cpu(full_map))
        return {
            "reconstructions": torch.cat(reconstructions, dim=0),
            "bgs": torch.cat(bgs, dim=0),
            "full_maps": full_maps,
        }

    def _combine_into_scene(self, chunk_est_dict: Dict, size_dict: Dict):
        recon_images = {}
        for chunk_type, chunk_est in chunk_est_dict.items():
            reconstructions = chunk_est["reconstructions"]
            bgs = chunk_est["bgs"]
            kernel_size = reconstructions.shape[-2], reconstructions.shape[-1]
            rr = rearrange(reconstructions - bgs, "(b n) c h w -> b (c h w) n", b=1, c=1)
            output_size = size_dict[chunk_type]
            recon_no_bg = F.fold(
                rr, output_size=output_size, kernel_size=kernel_size, stride=self.slen
            )
            n_tiles_h = (output_size[0] - 2 * self.bp) / self.slen
            bgs = rearrange(bgs, "(b nh nw) c h w -> b nh nw c h w", b=1, nh=n_tiles_h)
            bgs[:, :-1, :, :, -self.bp :, :] = 0.0
            bgs[:, :, :-1, :, :, -self.bp :] = 0.0
            bg_rearranged = rearrange(bgs, "b nh nw c h w -> b (c h w) (nh nw)", b=1, c=1)
            bg_all = F.fold(
                bg_rearranged, output_size=output_size, kernel_size=kernel_size, stride=self.slen
            )
            recon = recon_no_bg + bg_all
            recon_images[chunk_type] = recon

        ## Assemble the final image
        return torch.cat(
            (
                torch.cat(recon_images["main"], recon_images["right"][:, :, :, self.bp :], dim=2),
                torch.cat(
                    recon_images["bottom"][:, :, self.bp :, :],
                    recon_images["bottom_right"][:, :, self.bp :, self.bp :],
                    dim=2,
                ),
            ),
            dim=3,
        )

    def _combine_full_maps(self, chunk_est_dict: Dict):
        params = {}
        for chunk_type, chunk_est in chunk_est_dict.items():
            full_maps = chunk_est["full_maps"]
            biases = self.biases[chunk_type]
            plocs = []
            # Get new locations
            for i, bias in enumerate(biases):
                max_sources = full_maps[i]["locs"].shape[1]
                n_sources_i = full_maps[i]["n_sources"].unsqueeze(-1)

                bias_i = repeat(bias, "xy -> 1 max_sources xy", max_sources=max_sources)

                mask_i = torch.tensor(range(max_sources))
                mask_i = repeat(mask_i, "max_sources -> n max_sources", n=n_sources_i.shape[0])
                mask_i = mask_i < n_sources_i
                bias_i *= mask_i.unsqueeze(-1)

                plocs_i = full_maps[i]["plocs"] + bias_i
                plocs.append(plocs_i)
            plocs = torch.cat(plocs, dim=1)
            if params.get("plocs", False):
                params["plocs"] = torch.cat((params["plocs"], plocs))
            else:
                params["plocs"] = plocs

            for param_name in full_maps[0]:
                if param_name not in {"locs", "ploc", "n_sources"}:
                    tensors = torch.cat([full_map[param_name] for full_map in full_maps], dim=1)
                    if params.get(param_name, False):
                        params[param_name] = torch.cat((params[param_name], tensors), dim=1)
                    else:
                        params[param_name] = tensors
        return params


def reconstruct_img(
    encoder: Encoder, decoder: ImageDecoder, img: Tensor
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    img_ptiles = encoder.get_images_in_ptiles(img)

    with torch.no_grad():
        tile_map = encoder.max_a_post(img_ptiles)
        recon_image, _ = decoder.render_images(
            tile_map["n_sources"],
            tile_map["locs"],
            tile_map["galaxy_bools"],
            tile_map["galaxy_params"],
            tile_map["fluxes"],
            add_noise=False,
        )
        background = decoder.get_background(recon_image.shape[-2], recon_image.shape[-1]).unsqueeze(
            0
        )
        full_map = get_full_params_from_tiles(tile_map, decoder.tile_slen)
    return recon_image, background, full_map


def combine_chunks_into_scene(recon_chunks: List[Tensor], bgs: List[Tensor], slen: int):
    b = 1
    recon_minus_edge = fold_chunks_into_scene(
        torch.cat(recon_chunks[: -(2 * b)]), torch.cat(bgs[: -(2 * b)]), slen
    )
    kernel_size = recon_chunks[0].shape[-1]
    bp = kernel_size - slen
    last_h_chunk, last_w_chunk = recon_chunks[-2:]
    last_h_chunk = last_h_chunk[:, :, bp:, :]
    last_w_chunk = last_w_chunk[:, :, :, bp:]
    recon = torch.cat((recon_minus_edge, last_w_chunk), dim=3)
    recon = torch.cat((recon, last_h_chunk), dim=2)
    return recon


def fold_chunks_into_scene(recon_chunks: Tensor, bgs: Tensor, slen: int):
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


def combine_full_maps(
    full_maps: List[Dict[str, Tensor]], offsets: List[Tensor], chunk_slen: int
) -> Dict[str, Tensor]:
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

    # n_chunks_h = int(math.sqrt(len(full_maps)))
    # n_chunks_w = n_chunks_h
    # assert n_chunks_h * n_chunks_w == len(full_maps)

    # offsets_h = torch.tensor(range(n_chunks_h), device=params["locs"].device)
    # offsets_w = torch.tensor(range(n_chunks_w), device=params["locs"].device)
    # offsets = torch.cartesian_prod(offsets_h, offsets_w)
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
    # params["locs"][..., 0] = params["locs"][..., 0] / (chunk_slen * n_chunks_h)
    # params["locs"][..., 1] = params["locs"][..., 1] / (chunk_slen * n_chunks_w)

    return params
