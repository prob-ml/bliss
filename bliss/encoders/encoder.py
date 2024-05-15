"""Scripts to produce BLISS estimates on astronomical images."""

import math

import torch
from einops import pack, rearrange
from torch import Tensor, nn
from tqdm import tqdm

from bliss.catalog import TileCatalog
from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.render_tiles import get_images_in_tiles


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
        detection_encoder: DetectionEncoder,
        binary_encoder: BinaryEncoder | None = None,
        galaxy_encoder: GalaxyEncoder | None = None,
        n_images_per_batch: int = 10,
        n_rows_per_batch: int = 15,
    ):
        """Initializes Encoder.

        This module requires at least the `detection_encoder`. Other
        modules can be incorporated to add more information about the catalog,
        specifically whether an object is a galaxy or star (`binary_encoder`), or
        the latent parameter describing the shape of the galaxy `galaxy_encoder`.

        Args:
            detection_encoder: Module that takes padded tiles and returns the number
                of sources and locations per-tile.
            binary_encoder: Module that takes padded tiles and locations and
                returns a classification between stars and galaxies. Defaults to None.
            galaxy_encoder: Module that takes padded tiles and locations and returns the variational
                distribution of the latent variable determining the galaxy shape. Defaults to None.
            n_images_per_batch: How many images can be processed at a time on the GPU?
                If not specified, defaults to an amount known to fit on my GPU.
            n_rows_per_batch: How many vertical padded tiles can be processed at a time on the GPU?
                If not specified, defaults to an amount known to fit on my GPU.
        """
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.detection_encoder = detection_encoder
        self.binary_encoder = binary_encoder
        self.galaxy_encoder = galaxy_encoder

        self.n_images_per_batch = n_images_per_batch
        self.n_rows_per_batch = n_rows_per_batch

    def forward(self, x):
        raise NotImplementedError("Unavailable. Use .variational_mode() or .sample() instead.")

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
                - The output of DetectionEncoder.variational_mode()
                - 'galaxy_bools', 'star_bools', and 'galaxy_probs' from BinaryEncoder.
                - 'galaxy_params' from GalaxyEncoder.
        """
        n_tiles_h = (image.shape[2] - 2 * self.bp) // self.detection_encoder.tile_slen
        ptile_loader = self.make_ptile_loader(image, background, n_tiles_h)
        tile_map_list: list[dict[str, Tensor]] = []

        n_tiles_h = (image.shape[2] - 2 * self.bp) // self.detection_encoder.tile_slen
        n_tiles_w = (image.shape[3] - 2 * self.bp) // self.detection_encoder.tile_slen

        n1 = math.ceil(image.shape[0] / self.n_images_per_batch)
        n2 = math.ceil(n_tiles_h / self.n_rows_per_batch)
        total_n_ptiles = n1 * n2
        with torch.no_grad():
            for ptiles in tqdm(ptile_loader, desc="Encoding ptiles...", total=total_n_ptiles):
                out_ptiles = self._encode_ptiles(ptiles)
                tile_map_list.append(out_ptiles)

        tile_map = _collate(tile_map_list)

        return TileCatalog.from_flat_dict(
            self.detection_encoder.tile_slen,
            n_tiles_h,
            n_tiles_w,
            {k: v.squeeze(0) for k, v in tile_map.items()},
        )

    def make_ptile_loader(self, image: Tensor, background: Tensor, n_tiles_h: int):
        n_images = image.shape[0]
        img_bg, _ = pack([image, background], "b * h w")  # by default concatenate channels
        for start_b in range(0, n_images, self.n_images_per_batch):
            for row in range(0, n_tiles_h, self.n_rows_per_batch):
                end_b = start_b + self.n_images_per_batch
                end_row = row + self.n_rows_per_batch
                start_h = row * self.detection_encoder.tile_slen
                end_h = end_row * self.detection_encoder.tile_slen + 2 * self.bp
                img_bg_cropped = img_bg[start_b:end_b, :, start_h:end_h, :]
                image_ptiles = self._get_images_in_ptiles(img_bg_cropped)
                image_ptiles_flat = rearrange(image_ptiles, "b nth ntw c h w -> (b nth ntw) c h w")
                yield image_ptiles_flat.to(self.device)

    @property
    def bp(self) -> int:
        return self.detection_encoder.bp

    @property
    def device(self):
        return self._dummy_param.device

    def _encode_ptiles(self, flat_image_ptiles: Tensor):
        assert not self.detection_encoder.training
        tiled_params: dict[str, Tensor] = {}

        n_source_probs, locs_mean, locs_sd_raw = self.detection_encoder.encode_tiled(
            flat_image_ptiles
        )
        n_source_probs_inflated = rearrange(n_source_probs, "n -> n 1")  # for `TileCatalog`
        n_sources = n_source_probs.ge(0.5).long()
        tile_is_on = rearrange(n_sources, "np -> np 1")
        tiled_params.update({"n_sources": n_sources, "n_source_probs": n_source_probs_inflated})

        locs = locs_mean * tile_is_on
        locs_sd = locs_sd_raw * tile_is_on
        tiled_params.update({"locs": locs, "locs_sd": locs_sd})

        if self.binary_encoder is not None:
            assert not self.binary_encoder.training
            galaxy_probs = self.binary_encoder.encode_tiled(flat_image_ptiles, locs)
            galaxy_bools_flat = galaxy_probs.ge(0.5).float() * n_sources.float()
            galaxy_bools = rearrange(galaxy_bools_flat, "b -> b 1")
            tiled_params.update({"galaxy_bools": galaxy_bools})

            if self.galaxy_encoder is not None:
                assert not self.galaxy_encoder.training
                galaxy_params = self.galaxy_encoder.forward(flat_image_ptiles, locs)
                galaxy_params *= tile_is_on * galaxy_bools
                tiled_params.update({"galaxy_params": galaxy_params})

        return tiled_params

    def _get_images_in_ptiles(self, images):
        """Run get_images_in_ptiles with correct tile_slen and ptile_slen."""
        return get_images_in_tiles(
            images, self.detection_encoder.tile_slen, self.detection_encoder.ptile_slen
        )


def _collate(tile_map_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Combine multiple Tensors across dictionaries into a single dictionary."""
    assert tile_map_list  # not empty

    out: dict[str, Tensor] = {}
    for k in tile_map_list[0]:
        out[k] = torch.cat([d[k] for d in tile_map_list], dim=0)
    return out
