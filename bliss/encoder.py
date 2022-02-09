"""Scripts to produce BLISS estimates on survey images. Currently only SDSS is supported."""
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.location_encoder import (
    LocationEncoder,
    get_images_in_tiles,
    get_is_on_from_n_sources,
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
            of the catalog in tiles. Specifically, this dictionary comprises:
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
            galaxy_param_mean = self.galaxy_encoder.sample(image_ptiles, tile_map["locs"])
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


def get_star_bool(n_sources, galaxy_bool):
    assert n_sources.shape[0] == galaxy_bool.shape[0]
    assert galaxy_bool.shape[-1] == 1
    max_sources = galaxy_bool.shape[-2]
    assert n_sources.le(max_sources).all()
    is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
    is_on_array = is_on_array.view(*galaxy_bool.shape)
    return (1 - galaxy_bool) * is_on_array
