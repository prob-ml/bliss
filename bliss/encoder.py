"""Scripts to produce BLISS estimates on astronomical images."""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from bliss.catalog import (
    TileCatalog,
    TileCatalogSamples,
    get_images_in_tiles,
    get_is_on_from_n_sources,
)
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.models.location_encoder import LocationEncoder


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
        eval_mean_detections: Optional[float] = None,
        map_n_source_weights: Optional[Tuple[float, ...]] = None,
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
            eval_mean_detections: Optional. See LocationEncoder. Mean number of sources in each tile
                for test-time image. If provided, probabilities in location_encoder are adjusted.
            map_n_source_weights: Optional. See LocationEncoder. If specified, weights the argmax in
                MAP estimation of locations. Useful for raising/lowering the threshold for turning
                sources on/off.
        """
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.location_encoder = location_encoder
        self.binary_encoder = binary_encoder
        self.galaxy_encoder = galaxy_encoder
        self.eval_mean_detections = eval_mean_detections

        if map_n_source_weights is None:
            map_n_source_weights_tnsr = torch.ones(self.location_encoder.max_detections + 1)
        else:
            map_n_source_weights_tnsr = torch.tensor(map_n_source_weights)
        self.register_buffer("map_n_source_weights", map_n_source_weights_tnsr, persistent=False)

    def forward(self, x):
        raise NotImplementedError(
            ".forward() method for Encoder not available. Use .max_a_post() or .sample()."
        )

    def sample(
        self, image: Tensor, background: Tensor, n_samples: Optional[int] = None
    ) -> TileCatalogSamples:
        var_params = self.location_encoder.encode(
            image, background, eval_mean_detections=self.eval_mean_detections
        )
        if n_samples is None:
            assert isinstance(self.map_n_source_weights, Tensor)
            tile_map = self.location_encoder.max_a_post(
                var_params, n_source_weights=self.map_n_source_weights
            )
            tile_catalog_samples = TileCatalogSamples.unflatten_sample_and_batch_dim(1, tile_map)
        else:
            tile_catalog_samples = self.location_encoder.sample(var_params, n_samples)
        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            locs = tile_catalog_samples.locs.reshape(-1, *tile_catalog_samples.shape[2:], 2)
            image = image.expand(locs.shape[0], -1, -1, -1)
            background = image.expand(locs.shape[0], -1, -1, -1)
            galaxy_probs = self.binary_encoder.forward(image, background, locs)
            galaxy_probs = galaxy_probs.reshape(*tile_catalog_samples.shape, 1)
            galaxy_probs *= tile_catalog_samples.is_on_array.unsqueeze(-1)
            if n_samples is None:
                galaxy_bools = (
                    galaxy_probs > 0.5
                ).float() * tile_catalog_samples.is_on_array.unsqueeze(-1)
            else:
                galaxy_bools = (
                    torch.rand_like(galaxy_probs) <= galaxy_probs
                ) * tile_catalog_samples.is_on_array.unsqueeze(-1)
            star_bools = get_star_bools(tile_catalog_samples.n_sources, galaxy_bools)
            tile_catalog_samples.update(
                {
                    "galaxy_bools": galaxy_bools,
                    "star_bools": star_bools,
                    "galaxy_probs": galaxy_probs,
                }
            )

        if self.galaxy_encoder is not None:
            if n_samples is None:
                galaxy_params = self.galaxy_encoder.max_a_post(image, background, locs)
            else:
                galaxy_params = self.galaxy_encoder.sample(image, background, locs)
            galaxy_params = galaxy_params.reshape(*tile_catalog_samples.shape, -1)
            galaxy_params *= (
                tile_catalog_samples.is_on_array.unsqueeze(-1)
                * tile_catalog_samples["galaxy_bools"]
            )
            tile_catalog_samples.update({"galaxy_params": galaxy_params})
        return tile_catalog_samples

    def max_a_post(self, image: Tensor, background: Tensor) -> TileCatalog:
        tile_samples = self.sample(image, background, n_samples=None)
        return tile_samples.flatten_sample_and_batch_dim()

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
