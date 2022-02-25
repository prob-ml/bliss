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
from bliss.models.prior import ImagePrior


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
        prior: Optional[ImagePrior] = None,
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
        self.prior = prior

    def forward(self, x):
        raise NotImplementedError(
            ".forward() method for Encoder not available. Use .max_a_post() or .sample()."
        )

    def sample(self, n_samples: int, image: Tensor, background: Tensor) -> Dict[str, Tensor]:
        log_image = subtract_bg_and_log_transform(image, background)
        log_image_ptiles = self.get_images_in_ptiles(log_image)
        del log_image
        var_params = self.location_encoder.encode(log_image_ptiles)
        tile_samples = self.location_encoder.sample(var_params, n_samples)
        q_loc_tile_samples = self.location_encoder.log_prob(var_params, tile_samples)

        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            locs = rearrange(tile_samples["locs"], "ns n nth ntw s xy -> (ns n) nth ntw s xy")
            log_image_ptiles = log_image_ptiles.expand(n_samples, -1, -1, -1, -1, -1)
            galaxy_probs = self.binary_encoder(log_image_ptiles, locs)
            galaxy_probs = rearrange(
                galaxy_probs, "(ns n) nth ntw s 1 -> ns n nth ntw s 1", ns=n_samples
            )
            galaxy_probs *= tile_samples["is_on_array"]
            galaxy_bools = (galaxy_probs > torch.rand_like(galaxy_probs)).float()
            star_bools = get_star_bools(tile_samples["n_sources"], galaxy_bools)
            tile_samples.update(
                {
                    "galaxy_bools": galaxy_bools,
                    "star_bools": star_bools,
                    "galaxy_probs": galaxy_probs,
                }
            )
            q_binary = (
                galaxy_bools * galaxy_probs
                + star_bools * (1 - galaxy_probs)
                + (1 - tile_samples["is_on_array"])
            )
            q_binary = torch.log(q_binary)

        p_locs_and_binary = self.prior.log_prob(tile_samples)

        if ("galaxy_bools" in tile_samples) and (self.galaxy_encoder is not None):
            del log_image_ptiles
            image_ptiles = self.get_images_in_ptiles(image - background)
            image_ptiles = image_ptiles.expand(n_samples, -1, -1, -1, -1, -1)
            galaxy_params, galaxy_pq_z = self.galaxy_encoder.encode(image_ptiles, locs)
            galaxy_params = rearrange(
                galaxy_params, "(ns n) nth ntw s d -> ns n nth ntw s d", ns=n_samples
            )
            galaxy_params *= tile_samples["is_on_array"] * tile_samples["galaxy_bools"]
            galaxy_pq_z = rearrange(
                galaxy_pq_z, "(ns n) nth ntw s -> ns n nth ntw s d", ns=n_samples
            )
            galaxy_pq_z *= tile_samples["is_on_array"] * tile_samples["galaxy_bools"]
            tile_samples.update({"galaxy_params": galaxy_params})

        p_minus_q = (p_locs_and_binary + galaxy_pq_z) - q_loc_tile_samples - q_binary

        return tile_samples, p_minus_q

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
        log_image = subtract_bg_and_log_transform(image, background)
        log_image_ptiles = self.get_images_in_ptiles(log_image)
        del log_image
        var_params = self.location_encoder.encode(log_image_ptiles)
        tile_map = self.location_encoder.max_a_post(var_params)

        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            galaxy_probs = self.binary_encoder(log_image_ptiles, tile_map["locs"])
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
