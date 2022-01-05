"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from bliss.models.binary import BinaryEncoder
from bliss.models.location_encoder import (
    LocationEncoder,
    get_full_params_from_tiles,
    get_images_in_tiles,
    get_is_on_from_n_sources,
)
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.galaxy_net import OneCenteredGalaxyDecoder


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
            prob_galaxy = prob_galaxy.view(-1, 1, 1)
            prob_galaxy *= tile_map["is_on_array"]
            galaxy_bool = (prob_galaxy > 0.5).float() * tile_map["is_on_array"]
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
                        full_map = get_full_params_from_tiles(
                            tile_map, self.image_encoder.tile_slen
                        )
                        full_map = {k: v.cpu() for k, v in full_map.items()}
                        # delete parameters we stopped using so we have enough GPU space.
                        if "cuda" in self.device.type:
                            del pchunk
                            torch.cuda.empty_cache()

                        for k, param in full_map_scene.items():
                            full_map_scene[k] = torch.cat(param, full_map[k])

                        # update progress bar
                        pbar.update(1)

        return full_map_scene

    @property
    def device(self):
        return self._dummy_param.device


def get_star_bool(n_sources, galaxy_bool):
    assert n_sources.shape[0] == galaxy_bool.shape[0]
    assert galaxy_bool.shape[-1] == 1
    max_sources = galaxy_bool.shape[-2]
    assert n_sources.le(max_sources).all()
    is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
    is_on_array = is_on_array.view(*galaxy_bool.shape)
    return (1 - galaxy_bool) * is_on_array
