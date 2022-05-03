"""Scripts to produce BLISS estimates on astronomical images."""
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
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
        map_n_source_weights: Optional[Tuple[float, ...]] = None,
        batch_size: Optional[int] = None,
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
            map_n_source_weights: Optional. See LocationEncoder. If specified, weights the argmax in
                MAP estimation of locations. Useful for raising/lowering the threshold for turning
                sources on/off.
            batch_size: How many padded tiles can be rendered at a time on the GPU?
                If not specified, defaults to an amount known to fit on my GPU.
        """
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.location_encoder = location_encoder
        self.binary_encoder = binary_encoder
        self.galaxy_encoder = galaxy_encoder

        if map_n_source_weights is None:
            map_n_source_weights_tnsr = torch.ones(self.location_encoder.max_detections + 1)
        else:
            map_n_source_weights_tnsr = torch.tensor(map_n_source_weights)

        self.batch_size = batch_size if batch_size is not None else 75**2 + 500 * 5
        self.register_buffer("map_n_source_weights", map_n_source_weights_tnsr, persistent=False)

    def forward(self, x):
        raise NotImplementedError("Unavailable. Use .variational_mode() or .sample() instead.")

    def sample(self, image_ptiles, n_samples):
        raise NotImplementedError("Sampling from Encoder not yet available.")

    def variational_mode(self, image: Tensor, background: Tensor) -> TileCatalog:
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
                - The output of LocationEncoder.variational_mode()
                - 'galaxy_bools', 'star_bools', and 'galaxy_probs' from BinaryEncoder.
                - 'galaxy_params' from GalaxyEncoder.
        """
        n_tiles_h = (image.shape[2] - 2 * self.border_padding) // self.location_encoder.tile_slen
        n_tiles_w = (image.shape[3] - 2 * self.border_padding) // self.location_encoder.tile_slen
        ptile_loader = self._make_ptile_loader(image, background, n_tiles_h, n_tiles_w)
        tile_map_list: List[Dict[str, Tensor]] = []
        with torch.no_grad():
            for ptiles in tqdm(ptile_loader, desc="Encoding ptiles"):
                assert isinstance(ptiles, Tensor)
                out_ptiles = self._encode_ptiles(ptiles)
                tile_map_list.append(out_ptiles)
        tile_map_dict = self._collate(tile_map_list)
        return TileCatalog.from_flat_dict(
            self.location_encoder.tile_slen, n_tiles_h, n_tiles_w, tile_map_dict
        )

    def _make_ptile_loader(self, image: Tensor, background: Tensor, n_tiles_h: int, n_tiles_w: int):
        img_bg = torch.cat((image, background), dim=1).to(self.device)
        n_rows_per_batch = self.batch_size // n_tiles_w
        for row in range(0, n_tiles_h, n_rows_per_batch):
            end_row = row + n_rows_per_batch
            start_h = row * self.location_encoder.tile_slen
            end_h = end_row * self.location_encoder.tile_slen + 2 * self.border_padding
            img_bg_cropped = img_bg[:, :, start_h:end_h, :]
            image_ptiles = get_images_in_tiles(
                img_bg_cropped, self.location_encoder.tile_slen, self.location_encoder.ptile_slen
            )
            image_ptiles = image_ptiles.view(-1, *image_ptiles.shape[-3:])
            yield image_ptiles

    def _encode_ptiles(self, image_ptiles: Tensor):
        assert isinstance(self.map_n_source_weights, Tensor)
        dist_params = self.location_encoder.encode(image_ptiles)
        tile_map_dict = self.location_encoder.variational_mode(
            dist_params, n_source_weights=self.map_n_source_weights
        )
        locs = tile_map_dict["locs"]
        n_sources = tile_map_dict["n_sources"]
        is_on_array = get_is_on_from_n_sources(n_sources, self.location_encoder.max_detections)
        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            galaxy_probs = self.binary_encoder.forward(image_ptiles, locs)
            galaxy_probs *= is_on_array.unsqueeze(-1)
            galaxy_bools = (galaxy_probs > 0.5).float() * is_on_array.unsqueeze(-1)
            star_bools = get_star_bools(n_sources, galaxy_bools)
            tile_map_dict.update(
                {
                    "galaxy_bools": galaxy_bools,
                    "star_bools": star_bools,
                    "galaxy_probs": galaxy_probs,
                }
            )

        if self.galaxy_encoder is not None:
            galaxy_params = self.galaxy_encoder.variational_mode(image_ptiles, locs)
            galaxy_params *= is_on_array.unsqueeze(-1) * galaxy_bools
            tile_map_dict.update({"galaxy_params": galaxy_params})

        return tile_map_dict

    @staticmethod
    def _collate(tile_map_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for k in tile_map_list[0]:
            out[k] = torch.cat([d[k] for d in tile_map_list], dim=0)
        return out

    def get_images_in_ptiles(self, images):
        """Run get_images_in_ptiles with correct tile_slen and ptile_slen."""
        return get_images_in_tiles(
            images, self.location_encoder.tile_slen, self.location_encoder.ptile_slen
        )

    @property
    def border_padding(self) -> int:
        return self.location_encoder.border_padding

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
