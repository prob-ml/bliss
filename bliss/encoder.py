"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
from typing import Dict, Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.location_encoder import (
    LocationEncoder,
    get_images_in_tiles,
    get_is_on_from_n_sources,
    subtract_bg_and_log_transform,
)


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
        z_threshold: float = 4.0,
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
        self.z_threshold = z_threshold

    def forward(self, x):
        raise NotImplementedError(
            ".forward() method for Encoder not available. Use .max_a_post() or .sample()."
        )

    def sample(self, image_ptiles, n_samples):
        raise NotImplementedError("Sampling from Encoder not yet available.")

    def max_a_post(self, image: Tensor, background: Tensor) -> Dict[str, Tensor]:
        """Get maximum a posteriori of catalog from image padded tiles.

        Note that, strictly speaking, this is not the true MAP of the variational
        distribution of the catalog.
        Rather, we use sequential estimation; the MAP of the locations is first estimated,
        then plugged-in to the binary and galaxy encoders. Thus, the binary and galaxy
        encoders are conditioned on the location MAP. The true MAP would require optimizing
        over the entire catalog jointly, but this is not tractable.

        Args:
            image: An astronomical image,
                with shape `n * n_bands * h * w`.
            background: Background associated with image,
                with shape `n * n_bands * h * w`.

        Returns:
            A dictionary of the maximum a posteriori
            of the catalog in tiles. Specifically, this dictionary comprises:
                - The output of LocationEncoder.max_a_post()
                - 'galaxy_bools', 'star_bools', and 'galaxy_probs' from BinaryEncoder.
                - 'galaxy_params' from GalaxyEncoder.
        """
        log_image = subtract_bg_and_log_transform(image, background, z_threshold=self.z_threshold)
        log_image_ptiles = self.get_images_in_ptiles(log_image)
        del log_image
        var_params = self.location_encoder.encode(log_image_ptiles)
        tile_map = self.location_encoder.max_a_post(var_params)

        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            centered_ptiles = self.binary_encoder.get_images_in_tiles(image, background, tile_map["locs"], z_threshold=self.z_threshold)
            galaxy_probs = self.binary_encoder.forward(centered_ptiles)
            batch_size, n_tiles_h, n_tiles_w, max_sources, _ = tile_map["locs"].shape
            galaxy_probs = rearrange(
                galaxy_probs,
                "(b nth ntw s) 1 -> b nth ntw s 1",
                b=batch_size,
                nth=n_tiles_h,
                ntw=n_tiles_w,
                s=max_sources,
            )
            galaxy_probs *= tile_map["is_on_array"]
            galaxy_bools = (galaxy_probs > 0.5).float() * tile_map["is_on_array"]
            star_bools = get_star_bools(tile_map["n_sources"], galaxy_bools)
            tile_map.update(
                {
                    "galaxy_bools": galaxy_bools,
                    "star_bools": star_bools,
                    "galaxy_probs": galaxy_probs,
                }
            )

        if self.galaxy_encoder is not None:
            del log_image_ptiles
            image_ptiles = self.get_images_in_ptiles(image - background)
            galaxy_params = self.galaxy_encoder.max_a_post(image_ptiles, tile_map["locs"])
            galaxy_params *= tile_map["is_on_array"] * tile_map["galaxy_bools"]
            tile_map.update({"galaxy_params": galaxy_params})

        return tile_map

    def get_images_in_ptiles(self, images):
        """Run get_images_in_ptiles with correct tile_slen and ptile_slen."""
        return get_images_in_tiles(
            images, self.location_encoder.tile_slen, self.location_encoder.ptile_slen
        )

    @property
    def border_padding(self) -> int:
        return self.location_encoder.border_padding

    @property
    def tile_slen(self) -> int:
        return self.location_encoder.tile_slen

    @property
    def device(self):
        return self._dummy_param.device


def get_star_bools(n_sources, galaxy_bools):
    assert n_sources.shape[0] == galaxy_bools.shape[0]
    assert galaxy_bools.shape[-1] == 1
    max_sources = galaxy_bools.shape[-2]
    assert n_sources.le(max_sources).all()
    is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
    is_on_array = is_on_array.view(*galaxy_bools.shape)
    return (1 - galaxy_bools) * is_on_array
