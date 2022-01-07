"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from tqdm import tqdm

from bliss.models.binary import BinaryEncoder
from bliss.models.location_encoder import (
    LocationEncoder,
    get_full_params_from_tiles,
    get_images_in_tiles,
    get_is_on_from_n_sources,
)
from bliss.models.galaxy_encoder import GalaxyEncoder


class Encoder(nn.Module):
    """Encodes astronomical image into variational parameters.

    This module takes an astronomical image, or specifically padded tiles
    of an astronomical image, and returns either samples from the variational
    distribution of the latent catalog of objects represented by that image.

    Alternatively, this module can also return a sequential 'maximum-a-posteriori'
    (though this is not the true MAP since estimation is done sequentially rather than
    for the joint distribution or parameters).

    Attributes:
        See the __init__ function for a description of the attributes, which are
        the submodules for specific components of the catalog.

    """

    def __init__(
        self,
        location_encoder: LocationEncoder,
        binary_encoder: Optional[BinaryEncoder] = None,
        galaxy_encoder: Optional[GalaxyEncoder] = None,
    ):
        """Initializes Encoder.

        This module requires at least the `location_encoder`. Other
        modules can be incorporated to add more information about the catalog,
        specifically whether an object is a galaxy or star (`binary_encoder`), or
        the latent parameter describing the shape of the galaxy `galaxy_encoder`.

        Args:
            location_encoder: Module that takes padded tiles and returns the number
                of sources and locations per-tile.
            binary_encoder: Module that takes padded tiles and locations and
                returns a classification between stars and galaxies. Defaults to None.
            galaxy_encoder: Module that takes padded tiles and locations and returns the variational
                distribution of the latent variable determining the galaxy shape. Defaults to None.
        """
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.location_encoder = location_encoder
        self.binary_encoder = binary_encoder
        self.galaxy_encoder = galaxy_encoder

    def forward(self, x):
        raise NotImplementedError(
            ".forward() method for Encoder not available. Use .max_a_post() or .sample()."
        )

    def sample(self, image_ptiles, n_samples):
        raise NotImplementedError("Sampling from Encoder not yet available.")

    def max_a_post(self, image_ptiles: Tensor) -> Dict[str, Tensor]:
        """Get maximum a posteriori of catalog from image padded tiles.

        Note that, strictly speaking, this is not the true MAP of the variational
        distribution of the catalog.
        Rather, we use sequential estimation; the MAP of the locations is first estimated,
        then plugged-in to the binary and galaxy encoders. Thus, the binary and galaxy
        encoders are conditioned on the location MAP. The true MAP would require optimizing
        over the entire catalog jointly, but this is not tractable.

        Args:
            image_ptiles: A tensor of padded image tiles,
                with shape `n_ptiles * n_bands * h * w`.

        Returns:
            A dictionary of the maximum a posteriori
            of the catalog. Specifically, this dictionary comprises:
            - The output of LocationEncoder.max_a_post()
            - 'galaxy_bool', 'star_bool', and 'prob_galaxy' from BinaryEncoder.
            - 'galaxy_param' from GalaxyEncoder.
        """
        var_params = self.location_encoder.encode(image_ptiles)
        tile_map = self.location_encoder.max_a_post(var_params)

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
        """Run get_images_in_ptiles with correct tile_slen and ptile_slen."""
        return get_images_in_tiles(
            images, self.location_encoder.tile_slen, self.location_encoder.ptile_slen
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
        assert scene.shape[1] == self.location_encoder.n_bands == 1, "Only 1 band supported"
        h, w = scene.shape[-2], scene.shape[-1]
        bp = self.location_encoder.border_padding
        ihic = h // clen + 1 if not testing else 1  # height in chunks
        iwic = w // clen + 1 if not testing else 1  # width in chunks
        self.to(device)

        # tiles
        tile_slen = self.location_encoder.tile_slen

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
                            tile_map, self.location_encoder.tile_slen
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
