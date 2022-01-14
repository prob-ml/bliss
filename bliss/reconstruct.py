import torch
import math
from torch.nn import functional as F
from einops import rearrange
from tqdm import tqdm

from bliss.encoder import Encoder
from bliss.models.decoder import ImageDecoder
from bliss.models.location_encoder import get_params_in_batches


def reconstruct_scene_at_coordinates(encoder, decoder, img, h, w, scene_length, slen=80, bp=24):
    # Adjust bp so that we get an even number of chunks in scene
    # kernel_size = slen + bp * 2
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
    # h_padded = h - bp
    # w_padded = w - bp
    scene = img[
        :, :, h_padded : (h_padded + adj_scene_length), w_padded : (w_padded + adj_scene_length)
    ]
    assert scene.shape[2] == scene.shape[3] == adj_scene_length

    recon = reconstruct_scene(encoder, decoder, scene, slen, bp)

    ## Get reconstruction at coordinates
    return recon[
        :,
        :,
        (h - h_padded) : (h - h_padded + scene_length),
        (w - w_padded) : (w - w_padded + scene_length),
    ]
    # Add more


def reconstruct_scene(encoder, decoder, img_of_interest, slen=80, bp=24):
    chunks = split_scene_into_chunks(img_of_interest, slen, bp)
    reconstructions = []
    bgs = []
    for chunk in tqdm(chunks):
        recon, bg = reconstruct_img(encoder, decoder, chunk.unsqueeze(0).cuda())
        reconstructions.append(recon.cpu())
        bgs.append(bg.cpu())
    reconstructions = torch.cat(reconstructions, dim=0)
    bgs = torch.cat(bgs, dim=0)
    return combine_chunks_into_scene(reconstructions, bgs, slen)


def split_scene_into_chunks(scene, slen, bp):
    kernel_size = slen + bp * 2
    chunks = F.unfold(scene, kernel_size=kernel_size, stride=slen)
    chunks = rearrange(
        chunks,
        "b (c h w) n -> (b n) c h w",
        c=scene.shape[1],
        h=kernel_size,
        w=kernel_size,
    )
    return chunks


def reconstruct_img(encoder: Encoder, decoder: ImageDecoder, img):
    img_ptiles = encoder.get_images_in_ptiles(img)

    with torch.no_grad():
        tile_map = encoder.max_a_post(img_ptiles)
        tile_map = get_params_in_batches(tile_map, img.shape[0])
        recon_image, _ = decoder.render_images(
            tile_map["n_sources"],
            tile_map["locs"],
            tile_map["galaxy_bool"],
            tile_map["galaxy_param"],
            tile_map["fluxes"],
            add_noise=False,
        )
        background = decoder.get_background(recon_image.shape[-1]).unsqueeze(0)
    return recon_image, background


def combine_chunks_into_scene(recon_chunks, bgs, slen):
    kernel_size = recon_chunks.shape[-1]
    bp = kernel_size - slen
    rr = rearrange(recon_chunks - bgs, "(b n) c h w -> b (c h w) n", b=1, c=1)
    # rr = rearrange(recon_chunks, "(b n) c h w -> b (c h w) n", b=1, c=1)
    n_tiles_h = int(math.sqrt(recon_chunks.shape[0]))
    output_size = kernel_size + (n_tiles_h - 1) * slen
    rrr = F.fold(rr, output_size=output_size, kernel_size=kernel_size, stride=slen)
    # rfinal = rrr + 865
    # We now need to add back in the background; we zero out the right and bottom borders
    # to avoid double-counting
    bgs = rearrange(bgs, "(b nh nw) c h w -> b nh nw c h w", b=1, nh=n_tiles_h)
    bgs[:, :-1, :, :, -bp:, :] = 0.0
    bgs[:, :, :-1, :, :, -bp:] = 0.0
    bgr = rearrange(bgs, "b nh nw c h w -> b (c h w) (nh nw)", b=1, c=1)
    print(bgr.shape)
    print(rr.shape)
    bgrrr = F.fold(bgr, output_size=output_size, kernel_size=kernel_size, stride=slen)
    # rfinal = rrr
    return rrr + bgrrr
