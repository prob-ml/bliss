"""File to produce BLISS estimates on survey images. Currently only SDSS is supported."""
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bliss.datasets import sdss
from bliss.models import encoder
from bliss.models.binary import BinaryEncoder
from bliss.models.encoder import get_full_params, get_is_on_from_n_sources, get_star_bool
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase


def predict_on_image(
    image: torch.Tensor,
    image_encoder: encoder.ImageEncoder,
    galaxy_encoder: GalaxyEncoder = None,
    binary_encoder: BinaryEncoder = None,
):
    """This function takes in a single image and outputs the prediction from trained models.

    Prediction requires counts/locations provided by a trained `image_encoder` so this is
    a required argument. Note that prediction is done on a central square of the image
    corresponding to a border of size `image_encoder.border_padding`.

    Args:
        image: Tensor of shape (1, n_bands, slen, slen) where slen-2*border_padding <= 300.
        image_encoder: Trained ImageEncoder model. Assumed to be in correct device already.
        galaxy_encoder: Trained GalaxyEncoder model. Assumed to be in correct device already.
        binary_encoder: Trained BinaryEncoder model. Assumed to be in correct device already.

    Returns:
        tile_map: Dictionary containing MAP estimates for parameters of sources in each tile.
        full_map: Dictionary containing MAP estimates for parameters of sources on full image.
        var_params: Dictionary containing tensors of variational parameters corresponding
            to each tile in `image`. The variational parameters include `prob_galaxy`,
            `prob_n_sources`, `loc_mean`, `loc_logvar`, etc.
    """
    # prepare and check consistency
    assert not image_encoder.training
    assert len(image.shape) == 4
    assert image.shape[0] == 1
    assert image.shape[1] == image_encoder.n_bands
    assert image_encoder.max_detections == 1

    # prepare dimensions
    h, w = image.shape[-2], image.shape[-1]
    bp = image_encoder.border_padding

    # get padded tiles.
    ptiles = image_encoder.get_images_in_tiles(image)

    # get MAP estimates
    tile_n_sources = image_encoder.tile_map_n_sources(ptiles)
    tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, 1).reshape(1, -1, 1, 1)
    tile_map = image_encoder.tile_map_estimate(image)

    # get var_params in tiles (excluding galaxy params)
    var_params = image_encoder(ptiles, tile_n_sources)

    if galaxy_encoder is not None:
        # get galaxy params per tile
        assert not galaxy_encoder.training
        assert image.shape[1] == galaxy_encoder.n_bands
        assert image_encoder.border_padding == galaxy_encoder.border_padding
        assert galaxy_encoder.image_decoder.max_sources == 1
        assert image_encoder.tile_slen == galaxy_encoder.tile_slen

        # TODO: Do we need to zero out tiles with stars?
        galaxy_param_mean = galaxy_encoder(ptiles, tile_map["locs"])
        latent_dim = galaxy_param_mean.shape[-1]
        galaxy_param_mean = galaxy_param_mean.reshape(1, -1, 1, latent_dim)
        galaxy_param_mean *= tile_is_on_array
        var_params["galaxy_param_mean"] = galaxy_param_mean
        tile_map["galaxy_params"] = galaxy_param_mean

    if binary_encoder is not None:
        # get classification params per tile
        assert not binary_encoder.training
        assert image.shape[1] == binary_encoder.n_bands
        prob_galaxy = (
            binary_encoder(ptiles, tile_map["locs"]).reshape(1, -1, 1, 1) * tile_is_on_array
        )
        galaxy_bool = (prob_galaxy > 0.5).float().reshape(1, -1, 1, 1) * tile_is_on_array
        star_bool = get_star_bool(tile_map["n_sources"], galaxy_bool).reshape(1, -1, 1, 1)
        var_params["galaxy_bool"] = galaxy_bool
        tile_map["galaxy_bool"] = galaxy_bool
        var_params["star_bool"] = star_bool
        tile_map["star_bool"] = star_bool
        tile_map["prob_galaxy"] = prob_galaxy
        var_params["prob_galaxy"] = prob_galaxy

    # full parameters on chunk
    full_map = get_full_params(tile_map, h - 2 * bp, w - 2 * bp)

    return tile_map, full_map, var_params


def predict_on_scene(
    clen: int,
    scene: torch.Tensor,
    image_encoder: encoder.ImageEncoder,
    binary_encoder: BinaryEncoder = None,
    galaxy_encoder: GalaxyEncoder = None,
    device="cpu",
    testing=False,
):
    """Perform predictions chunk-by-chunk when image is larger than 300x300 pixels.

    The scene will be divided up into chunks of side length `clen`. Prediction will be
    done in every part of the scene except for a border of length `image_encoder.border_padding`.
    To be more specific, any sources with centroids (x0, y0) satisfying any of the following
    conditions: ``0 < x0 < bp``, ``w - bp < x0 < w``, ``0 < y0 < bp``, ``h - bp < y0 < h``
    will NOT be detected by our models.

    Args:
        clen: Dimensions of (unpadded) chunks we want to extract from scene.
        scene: Tensor of shape (1, n_bands, h, w) containing image of scene we will make
            predictions.
        device: Device where each model is currently and where padded chunks will be moved.
        image_encoder: Trained ImageEncoder model.
        galaxy_encoder: Trained GalaxyEncoder model.
        binary_encoder: Trained BinaryEncoder model.
        testing: Whether we are unit testing and we only want to run 1 chunk.

    Returns:
        results: List containing the results of prediction on each chunk, i.e. tuples of
            `tile_map, full_map, var_params` as returned by `predict_on_image`.
    """
    assert len(scene.shape) == 4
    assert scene.shape[0] == 1
    h, w = scene.shape[-2], scene.shape[-1]
    bp = image_encoder.border_padding
    ihic = h // clen if not testing else 1  # height in chunks
    iwic = w // clen if not testing else 1  # width in chunks

    # where to collect results.
    var_params = []
    locs = torch.tensor([])
    galaxy_bool = torch.tensor([])
    prob_galaxy = torch.tensor([])

    with torch.no_grad():
        with tqdm(total=ihic * iwic) as pbar:
            for i in range(ihic):
                for j in range(iwic):
                    x1, y1 = i * clen + bp, j * clen + bp
                    pchunk = scene[:, :, y1 - bp : y1 + clen + bp, x1 - bp : x1 + clen + bp]
                    assert pchunk.shape[-1] == pchunk.shape[-2] == clen + 2 * bp
                    pchunk = pchunk.to(device)

                    _, est_params, vparams = predict_on_image(
                        pchunk, image_encoder, galaxy_encoder, binary_encoder
                    )

                    est_locs = est_params["locs"].cpu().reshape(-1, 2)
                    est_gbool = est_params["galaxy_bool"].cpu().reshape(-1, 1)
                    est_pgalaxy = est_params["prob_galaxy"].cpu().reshape(-1, 1)

                    # locations in pixels consistent with full scene.
                    x = est_locs[:, 1].reshape(-1, 1) * clen + x1 - 0.5
                    y = est_locs[:, 0].reshape(-1, 1) * clen + y1 - 0.5

                    est_locs = torch.hstack((x, y)).reshape(-1, 2)

                    # concatenate to obtain tensors on full image.
                    locs = torch.cat((locs, est_locs))
                    galaxy_bool = torch.cat((galaxy_bool, est_gbool))
                    prob_galaxy = torch.cat((prob_galaxy, est_pgalaxy))

                    # save variational parameters
                    vparams = {key: value.cpu() for key, value in vparams.items()}
                    var_params.append(vparams)

                    # delete extra stuff in GPU and clear cache for next iteration.
                    if "cuda" in device:
                        del est_params
                        torch.cuda.empty_cache()

                    # update progress bar
                    pbar.update(1)

    full_map = {"locs": locs, "galaxy_bool": galaxy_bool, "prob_galaxy": prob_galaxy}

    return vparams, full_map


def predict(cfg: DictConfig):
    bands = list(cfg.predict.bands)
    print("-" * 20 + " Predicting Configuration " + "-" * 20)
    print(OmegaConf.to_yaml(cfg.predict))
    assert isinstance(bands, list) and len(bands) == 1, "Only 1 band supported"

    # setup params from config
    device = cfg.predict.device
    clen = cfg.predict.clen
    testing = cfg.predict.testing

    # load images from SDSS for prediction.
    sdss_obj = sdss.SloanDigitalSkySurvey(**cfg.predict.sdss_kwargs)
    image = sdss_obj[0]["image"][0]
    image = rearrange(torch.from_numpy(image), "h w -> 1 1 h w")

    # load models.
    sleep_net = SleepPhase.load_from_checkpoint(cfg.predict.sleep_checkpoint)
    galaxy_encoder = GalaxyEncoder.load_from_checkpoint(cfg.predict.galaxy_checkpoint)
    binary_encoder = BinaryEncoder.load_from_checkpoint(cfg.predict.binary_checkpoint)

    # move everything to specified GPU
    image_encoder = sleep_net.image_encoder.eval().to(device)
    galaxy_encoder = galaxy_encoder.to(device).eval()
    binary_encoder = binary_encoder.to(device).eval()

    var_params, _ = predict_on_scene(
        clen, image, image_encoder, binary_encoder, galaxy_encoder, device, testing
    )

    if cfg.predict.output_file is not None:
        torch.save(var_params, cfg.predict.output_file)
        print(f"Prediction saved to {cfg.predict.output_file}")
