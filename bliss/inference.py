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
    chunked_scene = ChunkedScene(scene, slen, bp)
    recon, map_scene = chunked_scene.reconstruct(encoder, decoder, device)
    assert recon.shape == scene.shape

    # recon, map_scene = reconstruct_scene(encoder, decoder, scene, slen=slen, device=device)

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
    # plocs = map_scene["plocs"]
    # plocs[..., 0] -= bp
    # plocs[..., 1] -= bp
    # map_scene["plocs"] = plocs

    return recon_at_coords, map_scene


class ChunkedScene:
    def __init__(self, scene: Tensor, slen: int, bp: int):
        """Split scenes into square chunks of side length `slen+bp*2` using `F.unfold`."""
        self.slen = slen
        self.bp = bp
        kernel_size = slen + bp * 2
        self.chunk_dict = {}
        self.size_dict = {}
        self.biases = {}

        n_chunks_h = (scene.shape[2] - (bp * 2)) // slen
        n_chunks_w = (scene.shape[3] - (bp * 2)) // slen
        self.chunk_dict["main"] = self._chunk_image(scene, (kernel_size, kernel_size), slen)
        self.size_dict["main"] = (n_chunks_h * slen + bp * 2, n_chunks_w * slen + bp * 2)

        offsets_h = torch.tensor(range(n_chunks_h))
        offsets_w = torch.tensor(range(n_chunks_w))
        offsets = torch.cartesian_prod(offsets_h, offsets_w)
        self.biases["main"] = offsets * self.slen

        # Get leftover chunks
        bottom_border_start = bp + slen * n_chunks_h - bp
        bottom_chunk_height = scene.shape[2] - bottom_border_start
        assert bottom_chunk_height >= bp * 2

        right_border_start = bp + slen * n_chunks_w - bp
        right_chunk_width = scene.shape[3] - right_border_start
        assert right_chunk_width >= bp * 2

        if bottom_chunk_height > bp * 2:
            bottom_border = scene[:, :, bottom_border_start:, : (right_border_start + 2 * bp)]
            self.chunk_dict["bottom"] = self._chunk_image(
                bottom_border, (bottom_chunk_height, kernel_size), slen
            )
            self.size_dict["bottom"] = bottom_border.shape[2:]
            self.biases["bottom"] = (
                torch.cartesian_prod(torch.tensor([n_chunks_h]), offsets_w) * self.slen
            )

        if right_chunk_width > bp * 2:
            right_border = scene[:, :, : (bottom_border_start + 2 * bp), right_border_start:]
            self.chunk_dict["right"] = self._chunk_image(
                right_border, (kernel_size, right_chunk_width), slen
            )
            self.size_dict["right"] = right_border.shape[2:]
            self.biases["right"] = (
                torch.cartesian_prod(offsets_h, torch.tensor([n_chunks_w])) * self.slen
            )

        if (bottom_chunk_height > bp * 2) and (right_chunk_width > bp * 2):
            bottom_right_border = scene[:, :, bottom_border_start:, right_border_start:]
            self.chunk_dict["bottom_right"] = bottom_right_border
            self.size_dict["bottom_right"] = bottom_right_border.shape[2:]
            self.biases["bottom_right"] = (
                torch.cartesian_prod(torch.tensor([n_chunks_h]), torch.tensor([n_chunks_w]))
                * self.slen
            )

    def _chunk_image(self, image, kernel_size, stride):
        chunks = F.unfold(image, kernel_size=kernel_size, stride=stride)
        return rearrange(
            chunks,
            "b (c h w) n -> (b n) c h w",
            c=image.shape[1],
            h=kernel_size[0],
            w=kernel_size[1],
        )

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

    def _combine_into_scene(self, chunk_est_dict: Dict):
        recon_images = {}
        recon_bgs = {}
        for chunk_type, chunk_est in chunk_est_dict.items():
            reconstructions = chunk_est["reconstructions"]
            bgs = chunk_est["bgs"]
            kernel_size = reconstructions.shape[-2], reconstructions.shape[-1]
            rr = rearrange(reconstructions - bgs, "(b n) c h w -> b (c h w) n", b=1, c=1)
            output_size = self.size_dict[chunk_type]
            recon_no_bg = F.fold(
                rr, output_size=output_size, kernel_size=kernel_size, stride=self.slen
            )
            n_tiles_h = math.ceil((output_size[0] - 2 * self.bp) / self.slen)
            bgs = rearrange(bgs, "(b nh nw) c h w -> b nh nw c h w", b=1, nh=n_tiles_h)
            bgs[:, :-1, :, :, -2 * self.bp :, :] = 0.0
            bgs[:, :, :-1, :, :, -2 * self.bp :] = 0.0
            bg_rearranged = rearrange(bgs, "b nh nw c h w -> b (c h w) (nh nw)", b=1, c=1)
            bg_all = F.fold(
                bg_rearranged, output_size=output_size, kernel_size=kernel_size, stride=self.slen
            )
            # recon = recon_no_bg + bg_all
            recon_images[chunk_type] = recon_no_bg
            recon_bgs[chunk_type] = bg_all

        ## Assemble the final image
        main = recon_images["main"]
        main_bg = recon_bgs["main"]
        right = recon_images.get("right", None)
        if right is None:
            top_of_image = main + main_bg
        else:
            right_bg = recon_bgs["right"]
            main[:, :, :, -2 * self.bp :] += right[:, :, :, : 2 * self.bp]
            main += main_bg
            top_of_image = torch.cat(
                (main, right[:, :, :, 2 * self.bp :] + right_bg[:, :, :, 2 * self.bp :]), dim=3
            )

        bottom = recon_images.get("bottom", None)
        if bottom is not None:
            bottom_bg = recon_bgs["bottom"]
            bottom_right = recon_images.get("bottom_right", None)
            if bottom_right is None:
                bottom_of_image = bottom
                bottom_of_image_bg = bottom_bg
            else:
                bottom_right_bg = recon_bgs["bottom_right"]
                bottom[:, :, :, -2 * self.bp :] += bottom_right[:, :, :, : 2 * self.bp]
                bottom_of_image = torch.cat((bottom, bottom_right[:, :, :, 2 * self.bp :]), dim=3)
                bottom_of_image_bg = torch.cat(
                    (bottom_bg, bottom_right_bg[:, :, :, 2 * self.bp :]), dim=3
                )
            top_of_image[:, :, -2 * self.bp :, :] += bottom_of_image[:, :, : 2 * self.bp, :]
            image = torch.cat(
                (
                    top_of_image,
                    bottom_of_image[:, :, 2 * self.bp :] + bottom_of_image_bg[:, :, 2 * self.bp :],
                ),
                dim=2,
            )
        else:
            image = top_of_image
        return image

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
            if params.get("plocs", None) is not None:
                params["plocs"] = torch.cat((params["plocs"], plocs), dim=1)
            else:
                params["plocs"] = plocs

            for param_name in full_maps[0]:
                if param_name not in {"locs", "plocs", "n_sources"}:
                    tensors = torch.cat([full_map[param_name] for full_map in full_maps], dim=1)
                    if params.get(param_name, None) is not None:
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
