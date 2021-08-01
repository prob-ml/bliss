"""File to produce BLISS estimates on survey images. Currently only SDSS is supported."""
import torch
from einops import rearrange
from omegaconf import DictConfig

from bliss.datasets import sdss
from bliss.models.encoder import get_full_params
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase


def prediction(image, image_encoder, galaxy_encoder):

    # prepare and check consistency
    assert len(image.shape) == 4
    assert image.shape[0] == 1
    assert image.shape[1] == image_encoder.n_bands == galaxy_encoder.n_bands
    image_encoder.eval()
    galaxy_encoder.eval()
    assert image_encoder.border_padding == galaxy_encoder.border_padding
    assert image_encoder.tile_slen == galaxy_encoder.tile_slen
    assert image_encoder.max_detections == galaxy_encoder.image_decoder.max_sources == 1
    h, w = image.shape[-2], image.shape[-1]
    bp = image_encoder.border_padding

    # get padded tiles.
    ptiles = image_encoder.get_images_in_tiles(image)

    # get MAP estimates
    tile_n_sources = image_encoder.tile_map_n_sources(ptiles)
    tile_map = image_encoder.tile_map_estimate(image)

    # get var_params in tiles (excluding galaxy params)
    var_params = image_encoder(ptiles, tile_n_sources)

    # get galaxy params per tile
    galaxy_param_mean = galaxy_encoder(ptiles, tile_map["locs"])

    # full parameters on chunk
    full_map = get_full_params(tile_map, h - 2 * bp, w - 2 * bp)

    # collect all parameters into dictionaries
    var_params["galaxy_param_mean"] = galaxy_param_mean
    tile_map["galaxy_params"] = galaxy_param_mean
    return var_params, tile_map, full_map


def predict(cfg: DictConfig):
    bands = list(cfg.predict.bands)
    assert isinstance(bands, list) and len(bands) == 1, "Only 1 band supported"

    sdss_obj = sdss.SloanDigitalSkySurvey(**cfg.predict.sdss_kwargs)
    sleep_net = SleepPhase.load_from_checkpoint(cfg.predict.sleep_checkpoint)
    galaxy_encoder = GalaxyEncoder.load_from_checkpoint(cfg.predict.galaxy_checkpoint)

    # load images from SDSS for prediction.
    image = sdss_obj[0]["image"][0]
    h, w = image.shape
    image = rearrange(torch.from_numpy(image), "h w -> 1 1 h w")

    # move everything to specified GPU
    sleep_net.to(cfg.predict.device)
    sleep_net.eval()
    image_encoder = sleep_net.image_encoder.eval()
    galaxy_encoder = galaxy_encoder.to(cfg.predict.device).eval()

    list_var_params = []
    clen = 300  # sdss image is too big so we need to chunk it.

    # number of chunks
    ihic = h // clen if not cfg.predict.testing else 1
    iwic = w // clen if not cfg.predict.testing else 1

    with torch.no_grad():
        for i in range(ihic):
            for j in range(iwic):
                chunk = image[:, :, i * clen : (i + 1) * clen, j * clen : (j + 1) * clen]
                chunk = chunk.to(cfg.predict.device)

                # predict!
                var_params, _, _ = prediction(chunk, image_encoder, galaxy_encoder)

                # put everything in the cpu before saving
                var_params = {key: value.cpu() for key, value in var_params.items()}
                list_var_params.append(var_params)

                # delete extra stuff in GPU and clear cache for next iteration.
                del chunk
                if "cuda" in cfg.predict.device:
                    torch.cuda.empty_cache()

    all_var_params = {}
    for var_params in list_var_params:
        for key in var_params:
            t1 = all_var_params.get(key, torch.tensor([]))
            t2 = var_params[key]
            all_var_params[key] = torch.cat((t1, t2))

    if cfg.predict.output_file is not None:
        torch.save(var_params, cfg.predict.output_file)
