import torch
from omegaconf import DictConfig

from bliss.datasets import sdss
from bliss import sleep

_models = [sleep.SleepPhase]
models = {cls.__name__: cls for cls in _models}


def predict(cfg: DictConfig):
    bands = list(cfg.predict.bands)
    assert isinstance(bands, list) and len(bands) == 1, "Only 1 band supported"

    sdss_obj = sdss.SloanDigitalSkySurvey(**cfg.predict.sdss)

    ## To regenerate this checkpoint:
    ## mode="train"
    ## model="sleep_galaxy_measure_stars1"
    ## dataset="default"
    ## optimizer.params.lr=1e-4
    ## training.n_epochs=201
    ## Save the output in cfg.predict.checkpoint
    sleep_net = sleep.SleepPhase.load_from_checkpoint(cfg.predict.checkpoint)

    # image for prediction from SDSS
    image = sdss_obj[0]["image"][bands[0]]
    h, w = image.shape
    image = torch.from_numpy(image.reshape(1, 1, h, w))

    # move everything to specified GPU
    sleep_net.to(cfg.predict.device)
    sleep_net.eval()
    image_encoder = sleep_net.image_encoder.eval()

    list_var_params = []
    clen = 200  # sdss image is too big so we need to chunk it.

    with torch.no_grad():
        for i in range(h // clen):
            for j in range(w // clen):
                chunk = image[
                    :, :, i * clen : (i + 1) * clen, j * clen : (j + 1) * clen
                ]
                chunk = chunk.to(cfg.predict.device)

                # tile the image
                ptiles = image_encoder.get_images_in_tiles(chunk)

                # use MAP estimate on n_sources and locs (for galaxy encoder)
                tile_n_sources = image_encoder.tile_map_n_sources(ptiles)
                tile_params = image_encoder.tile_map_estimate(chunk)

                # get var_params in tiles (excluding galaxy params)
                var_params = image_encoder.forward(ptiles, tile_n_sources)

                # get galaxy params per tile
                galaxy_param_mean, galaxy_param_var = sleep_net.forward_galaxy(
                    ptiles, tile_params["locs"]
                )

                # collect all parameters into one dictionary
                var_params["galaxy_param_mean"] = galaxy_param_mean
                var_params["gaalxy_param_var"] = galaxy_param_var

                # put everything in the cpu before saving
                var_params = {key: value.cpu() for key, value in var_params.items()}
                list_var_params.append(var_params)

                # delete extra stuff in GPU and clear cache for next iteration.
                del chunk
                del ptiles
                del tile_params
                del tile_n_sources
                if "cuda" in cfg.predict.device:
                    torch.cuda.empty_cache()

                # just for coverage so only run one index.
                if cfg.predict.testing:
                    break

    all_var_params = {}
    for var_params in list_var_params:
        for key in var_params:
            t1 = all_var_params.get(key, torch.tensor([]))
            t2 = var_params[key]
            all_var_params[key] = torch.cat((t1, t2))

    if cfg.predict.output_file is not None:
        torch.save(var_params, cfg.predict.output_file)
