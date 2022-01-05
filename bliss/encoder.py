"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from torch.tensor import Tensor
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bliss.datasets import sdss
from bliss.models.binary import BinaryEncoder
from bliss.models.location_encoder import (
    LocationEncoder,
    get_full_params,
    get_images_in_tiles,
    get_is_on_from_n_sources,
    get_params_in_batches,
)
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.galaxy_net import OneCenteredGalaxyDecoder
from bliss.sleep import SleepPhase


class Encoder(nn.Module):
    def __init__(
        self,
        image_encoder: LocationEncoder,
        binary_encoder: Optional[BinaryEncoder] = None,
        galaxy_encoder: Optional[GalaxyEncoder] = None,
        galaxy_decoder: Optional[OneCenteredGalaxyDecoder] = None,
    ):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.image_encoder = image_encoder
        self.binary_encoder = binary_encoder
        self.galaxy_encoder = galaxy_encoder
        self.galaxy_decoder = galaxy_decoder

    def forward(self, x):
        pass

    def sample(self, image_ptiles, n_samples):
        pass

    def max_a_post(self, image_ptiles):
        var_params = self.image_encoder.encode(image_ptiles)
        tile_map = self.image_encoder.max_a_post(var_params)

        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            prob_galaxy = self.binary_encoder(image_ptiles, tile_map["locs"])
            # print(prob_galaxy.shape)
            prob_galaxy = prob_galaxy.view(-1, 1, 1)
            prob_galaxy *= tile_map["is_on_array"]
            galaxy_bool = (prob_galaxy > 0.5).float() * tile_map["is_on_array"]
            # print(tile_map["n_sources"].shape)
            # print(galaxy_bool.shape)
            star_bool = get_star_bool(tile_map["n_sources"], galaxy_bool)
            tile_map.update(
                {
                    "galaxy_bool": galaxy_bool,
                    "star_bool": star_bool,
                    "prob_galaxy": prob_galaxy,
                }
            )

        if self.galaxy_encoder is not None:
            galaxy_param_mean = self.galaxy_encoder(image_ptiles, tile_map["locs"])
            latent_dim = galaxy_param_mean.shape[-1]
            galaxy_param_mean = galaxy_param_mean.reshape(1, -1, 1, latent_dim)
            galaxy_param_mean *= tile_map["is_on_array"] * tile_map["galaxy_bool"]
            tile_map.update({"galaxy_param": galaxy_param_mean})

        return tile_map

    def get_images_in_ptiles(self, images):
        return get_images_in_tiles(
            images, self.image_encoder.tile_slen, self.image_encoder.ptile_slen
        )

    def max_a_post_scene(
        self,
        scene: torch.Tensor,
        clen: int,
        device: torch.device = "cpu",
        testing=False,
    ):
        """Perform predictions chunk-by-chunk when image is larger than 300x300 pixels.

        The scene will be divided up into chunks of side length `clen`. Prediction will be
        done in every part of the scene except for a border of length
        `image_encoder.border_padding`.
        To be more specific, any sources with centroids (x0, y0) satisfying any of the following
        conditions: ``0 < x0 < bp``, ``w - bp < x0 < w``, ``0 < y0 < bp``, ``h - bp < y0 < h``
        will NOT be detected by our models.

        Args:
            clen: Dimensions of (unpadded) chunks we want to extract from scene.
            scene: Tensor of shape (1, n_bands, h, w) containing image of scene we will make
                predictions.
            device: Device where each model is currently and where padded chunks will be moved.
            testing: Whether we are unit testing and we only want to run 1 chunk.

        Returns:
            results: List containing the results of prediction on each chunk, i.e. tuples of
                `tile_map, full_map, var_params` as returned by `predict_on_image`.
        """
        assert len(scene.shape) == 4
        assert scene.shape[0] == 1
        assert scene.shape[1] == self.image_encoder.n_bands == 1, "Only 1 band supported"
        h, w = scene.shape[-2], scene.shape[-1]
        bp = self.image_encoder.border_padding
        ihic = h // clen + 1 if not testing else 1  # height in chunks
        iwic = w // clen + 1 if not testing else 1  # width in chunks
        self.to(device)

        # tiles
        tile_slen = self.image_encoder.tile_slen

        # where to collect results.
        # var_params = []
        full_map_scene = {
            "locs": torch.tensor([]),
            "galaxy_bool": torch.tensor([]),
            "prob_galaxy": torch.tensor([]),
            "fluxes": torch.tensor([]),
            "mags": torch.tensor([]),
        }

        with torch.no_grad():
            with tqdm(total=ihic * iwic) as pbar:
                for i in range(iwic):
                    for j in range(ihic):
                        x1, y1 = i * clen + bp, j * clen + bp

                        # the following two statements ensure divisibility by tile_slen
                        # of the resulting chunk near the edges.
                        if clen + bp > w - x1:
                            x1 = x1 + (w - x1 - bp) % tile_slen
                        if clen + bp > h - y1:
                            y1 = y1 + (h - y1 - bp) % tile_slen
                        pchunk = scene[:, :, y1 - bp : y1 + clen + bp, x1 - bp : x1 + clen + bp]
                        pchunk = pchunk.to(self.device)
                        image_ptiles = self.get_images_in_ptiles(pchunk)

                        tile_map = self.max_a_post(image_ptiles)
                        h_pchunk, w_pchunk = pchunk.shape[-2], pchunk.shape[-1]
                        full_map = get_full_params(tile_map, h_pchunk - 2 * bp, w_pchunk - 2 * bp)
                        full_map = {k: v.cpu() for k, v in full_map.items()}
                        # delete parameters we stopped using so we have enough GPU space.
                        if "cuda" in self.device.type:
                            del pchunk
                            torch.cuda.empty_cache()

                        for k in full_map_scene:
                            full_map_scene[k] = torch.cat(full_map_scene[k], full_map[k])

                        # update progress bar
                        pbar.update(1)

        return full_map_scene

    @property
    def device(self):
        return self._dummy_param.device

    # def _validate_image(self, image):
    #     # prepare and check consistency
    #     assert not self.image_encoder.training
    #     assert len(image.shape) == 4
    #     assert image.shape[0] == 1
    #     assert image.shape[1] == self.image_encoder.n_bands == 1
    #     assert self.image_encoder.max_detections == 1
    #     # binary prediction
    #     if self.binary_encoder is not None:
    #         assert not self.binary_encoder.training
    #         assert image.shape[1] == self.binary_encoder.n_bands
    #     # galaxy measurement predictions
    #     if self.galaxy_decoder is not None:
    #         assert not self.galaxy_encoder.training
    #         assert image.shape[1] == self.galaxy_encoder.n_bands
    #         assert self.image_encoder.border_padding == self.galaxy_encoder.border_padding
    #         assert self.image_encoder.tile_slen == self.galaxy_encoder.tile_slen


def predict(cfg: DictConfig):
    bands = list(cfg.predict.bands)
    print("-" * 20 + " Predicting Configuration " + "-" * 20)
    print(OmegaConf.to_yaml(cfg.predict))
    assert isinstance(bands, list) and len(bands) == 1, "Only 1 band supported"

    # setup params from config
    clen = cfg.predict.clen
    device = torch.device(cfg.predict.device)
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
    binary_encoder = binary_encoder.to(device).eval()
    galaxy_encoder = galaxy_encoder.to(device).eval()
    galaxy_decoder = sleep_net.image_decoder.galaxy_tile_decoder.galaxy_decoder.eval().to(device)

    predict_module = Predict(image_encoder, binary_encoder, galaxy_encoder, galaxy_decoder)
    var_params, _ = predict_module.predict_on_scene(clen, image, device, testing)

    if cfg.predict.output_file is not None:
        torch.save(var_params, cfg.predict.output_file)
        print(f"Prediction saved to {cfg.predict.output_file}")


def get_star_bool(n_sources, galaxy_bool):
    assert n_sources.shape[0] == galaxy_bool.shape[0]
    assert galaxy_bool.shape[-1] == 1
    max_sources = galaxy_bool.shape[-2]
    assert n_sources.le(max_sources).all()
    is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
    is_on_array = is_on_array.view(*galaxy_bool.shape)
    return (1 - galaxy_bool) * is_on_array
