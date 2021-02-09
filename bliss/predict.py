import torch
from omegaconf import DictConfig

from bliss.datasets import sdss
from bliss import sleep

_models = [sleep.SleepPhase]
models = {cls.__name__: cls for cls in _models}


def main(cfg: DictConfig):
    assert (
        isinstance(cfg.predict.band, list) and len(cfg.predict.band) == 1
    ), "Only 1 band supported"

    sdss_obj = sdss.SloanDigitalSkySurvey(**cfg.predict.sdss)
    sleep_net = sleep.SleepPhase.load_from_checkpoint(cfg.predict.checkpoint)

    # image for prediction from SDSS
    image = sdss_obj[0]["image"][cfg.predict.band[0]]
    h, w = image.shape
    image = image.reshape(1, 1, h, w)

    # move everything to specified GPU
    image.to(cfg.predict.device)
    sleep_net.to(cfg.predict.device)
    image_encoder = sleep_net.image_encoder

    # tile the image
    ptiles = image_encoder.get_images_in_tiles(image)

    # use MAP estimate on n_sources and locs (for galaxy encoder)
    tile_n_sources = image_encoder.tile_map_n_sources(ptiles)
    tile_params = image_encoder.tile_map_estimate(image)

    # get var_params in tiles (excluding galaxy params)
    var_params = image_encoder.forward(ptiles, tile_n_sources)

    # get galaxy params per tile
    galaxy_param_mean, galaxy_param_var = sleep_net.forward_galaxy(
        ptiles, tile_params["locs"]
    )

    # collect them
    var_params["galaxy_param_mean"] = galaxy_param_mean
    var_params["gaalxy_param_var"] = galaxy_param_var

    torch.save(var_params, cfg.predict.output_file)
